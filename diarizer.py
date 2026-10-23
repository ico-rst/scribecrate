"""
Speaker diarization via sherpa-onnx speaker embeddings (ONNX Runtime, no PyTorch).

Transcript segments are cut into speech chunks at word pauses and sentence
ends. Each chunk is embedded with a speaker model, the chunks are clustered
into speakers, and segments are split where the speaker changes.

https://github.com/hwdsl2/scribecrate

Copyright (C) 2026 Lin Song <linsongui@gmail.com>

This work is licensed under the MIT License
See: https://opensource.org/licenses/MIT
"""

import collections
import dataclasses
import logging
import os
import subprocess

import numpy as np
import sherpa_onnx

logger = logging.getLogger("whisper_server.diarizer")

# ---------------------------------------------------------------------------
# Model URL (GitHub releases from k2-fsa/sherpa-onnx)
# ---------------------------------------------------------------------------

# NVIDIA NeMo TitaNet-small speaker embedding model. Of the sherpa-onnx speaker
# models compared on English, German and Chinese recordings, it kept different
# speakers apart most reliably for both known and detected speaker counts.
_EMB_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/nemo_en_titanet_small.onnx"
)
_EMB_MODEL_REL = "nemo_en_titanet_small.onnx"

_PROVIDERS = ("cpu", "cuda")
_SAMPLE_RATE = 16000

# Tuned on two- to four-speaker English, German and Chinese recordings.
_CHUNK_GAP_SECONDS = 0.5     # cut segments at word gaps of at least this length
_SENTENCE_END = (".", "?", "!")  # and after words ending a sentence
_MIN_EMBED_SECONDS = 1.5     # chunks at least this long define the speakers
_MIN_SPEAKER_SECONDS = 3.0   # speakers with less speech join the nearest one
_MIN_TURN_SECONDS = 0.5      # shorter speaker changes inside a segment are ignored

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_extractor = None  # SpeakerEmbeddingExtractor instance
_num_speakers = -1  # exact speaker count, or -1 for threshold-based detection
_threshold = 0.45  # cosine-distance threshold between speaker centroids

_Chunk = collections.namedtuple("_Chunk", "segment start end words")


def _download_file(url: str, dest: str) -> None:
    """Download a file using curl (available in the container)."""
    logger.info("Downloading %s -> %s", url, dest)
    partial = dest + ".part"
    subprocess.run(
        ["curl", "-fSL", "--retry", "3", "--retry-delay", "2", "-o", partial, url],
        check=True,
    )
    # Never leave a truncated model where the next start would trust it.
    os.replace(partial, dest)


def _ensure_model(cache_dir: str) -> str:
    """Download the ONNX embedding model if not already present."""
    emb_path = os.path.join(cache_dir, _EMB_MODEL_REL)
    if not os.path.isfile(emb_path):
        _download_file(_EMB_MODEL_URL, emb_path)
    return emb_path


def load(
    cache_dir: str = "/var/lib/whisper",
    num_speakers: int = -1,
    cluster_threshold: float = 0.45,
    num_threads: int = 1,
    provider: str = "cpu",
) -> None:
    """Download the ONNX model if needed and initialize the embedding extractor.

    provider selects the ONNX Runtime execution provider: "cpu", or "cuda"
    (requires the CUDA build of sherpa-onnx in the :cuda image).
    """
    global _extractor, _num_speakers, _threshold

    # Reject invalid direct calls before downloads or native initialization.
    # bool is an int subclass, but is not a meaningful thread configuration.
    if type(num_threads) is not int or not 1 <= num_threads <= 2147483647:
        raise ValueError("num_threads must be an integer from 1 to 2147483647.")
    if provider not in _PROVIDERS:
        raise ValueError("provider must be one of: " + ", ".join(_PROVIDERS) + ".")

    emb_path = _ensure_model(cache_dir)

    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=emb_path, num_threads=num_threads, provider=provider,
    )
    if not config.validate():
        raise RuntimeError(
            "Diarization config validation failed. Check that the model file exists."
        )

    _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    # If the speaker count is unknown, use threshold-based detection.
    _num_speakers = num_speakers if num_speakers > 0 else -1
    _threshold = cluster_threshold
    logger.info(
        "Diarization model ready (num_speakers=%d, threshold=%.2f, num_threads=%d per model, provider=%s)",
        _num_speakers,
        _threshold,
        num_threads,
        provider,
    )


def is_loaded() -> bool:
    """Return True if the speaker embedding extractor is initialized."""
    return _extractor is not None


def _load_audio(audio_path: str, target_sr: int = 16000):
    """
    Load and resample audio file to mono float32 at target_sr using PyAV.
    Uses the same approach as faster-whisper's decode_audio (AudioResampler).
    PyAV is already installed as a dependency of faster-whisper and supports
    all formats that FFmpeg supports (mp3, ogg, m4a, webm, wav, flac, etc.).
    """
    import io

    import av

    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=target_sr,
    )

    raw_buffer = io.BytesIO()
    with av.open(audio_path, mode="r", metadata_errors="ignore") as container:
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                raw_buffer.write(resampled.to_ndarray())
        # Flush remaining buffered samples from the resampler
        for resampled in resampler.resample(None):
            if resampled.samples > 0:
                raw_buffer.write(resampled.to_ndarray())

    audio = np.frombuffer(raw_buffer.getvalue(), dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    return audio


def _split_into_chunks(segments):
    """Cut segments into chunks at word pauses and after sentence ends.

    Segments without word timestamps become a single chunk.
    """
    chunks = []
    for index, seg in enumerate(segments):
        words = list(seg.words or [])
        if not words:
            chunks.append(_Chunk(index, seg.start, seg.end, []))
            continue
        current = [words[0]]
        for prev, word in zip(words, words[1:]):
            if (word.start - prev.end >= _CHUNK_GAP_SECONDS
                    or prev.word.strip().endswith(_SENTENCE_END)):
                chunks.append(_Chunk(index, current[0].start, current[-1].end, current))
                current = []
            current.append(word)
        chunks.append(_Chunk(index, current[0].start, current[-1].end, current))
    return chunks


def _embed(audio, start: float, end: float):
    """Return the unit-length speaker embedding of an audio span, or None if too short."""
    samples = audio[max(int(start * _SAMPLE_RATE), 0):max(int(end * _SAMPLE_RATE), 0)]
    if samples.size == 0:
        return None
    stream = _extractor.create_stream()
    stream.accept_waveform(_SAMPLE_RATE, samples)
    stream.input_finished()
    if not _extractor.is_ready(stream):
        return None
    embedding = np.asarray(_extractor.compute(stream), dtype=np.float64)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else None


def _cluster(embeddings, durations, num_speakers: int, threshold: float):
    """Centroid-linkage clustering of unit embeddings, weighted by speech duration.

    Merging stops at the cosine-distance threshold, or at num_speakers. Both only
    count clusters with at least _MIN_SPEAKER_SECONDS of speech, so a stray chunk
    cannot claim a speaker of its own; small clusters then join the nearest one.
    Returns a label from 0 to (speakers - 1) for each row.
    """
    n = len(embeddings)
    sums = embeddings * durations[:, None]
    sizes = durations.astype(np.float64)
    owner = np.arange(n)  # cluster id (index of its first row) for each row
    active = np.ones(n, dtype=bool)
    # Short recordings cannot have two clusters with the minimum amount of
    # speech; treat every cluster as a speaker there.
    min_size = _MIN_SPEAKER_SECONDS if sizes.sum() >= 2 * _MIN_SPEAKER_SECONDS else 0.0

    centroids = embeddings.copy()
    distance = 1.0 - centroids @ centroids.T
    np.fill_diagonal(distance, np.inf)
    while active.sum() > 1:
        i, j = np.unravel_index(np.argmin(distance), distance.shape)
        large = sizes[active] >= min_size
        if num_speakers > 0:
            if sizes[i] >= min_size and sizes[j] >= min_size and large.sum() <= num_speakers:
                break
        elif distance[i, j] > threshold:
            break
        sums[i] += sums[j]
        sizes[i] += sizes[j]
        owner[owner == j] = i
        active[j] = False
        centroids[i] = sums[i] / max(np.linalg.norm(sums[i]), 1e-12)
        row = 1.0 - centroids @ centroids[i]
        row[~active] = np.inf
        row[i] = np.inf
        distance[i, :] = row
        distance[:, i] = row
        distance[j, :] = np.inf
        distance[:, j] = np.inf

    clusters = np.flatnonzero(active)
    speakers = [c for c in clusters if sizes[c] >= min_size] or list(clusters)
    speaker_centroids = centroids[speakers]
    target = {}
    for c in clusters:
        target[c] = speakers.index(c) if c in speakers else int(np.argmax(speaker_centroids @ centroids[c]))
    return np.array([target[c] for c in owner])


def _speaker_runs(labels, chunks):
    """Group a segment's chunk labels into runs, dropping changes shorter than _MIN_TURN_SECONDS."""
    runs = []  # [label, duration, words]
    for label, chunk in zip(labels, chunks):
        if runs and runs[-1][0] == label:
            runs[-1][1] += chunk.end - chunk.start
            runs[-1][2].extend(chunk.words)
        else:
            runs.append([label, chunk.end - chunk.start, list(chunk.words)])
    while len(runs) > 1:
        shortest = min(range(len(runs)), key=lambda r: runs[r][1])
        if runs[shortest][1] >= _MIN_TURN_SECONDS:
            break
        neighbor = shortest - 1 if shortest > 0 else 1
        first, second = sorted((shortest, neighbor))
        merged_label = runs[neighbor][0]
        runs[first:second + 1] = [[
            merged_label, runs[first][1] + runs[second][1], runs[first][2] + runs[second][2],
        ]]
        # Merging may join two runs of the same speaker.
        joined = []
        for run in runs:
            if joined and joined[-1][0] == run[0]:
                joined[-1][1] += run[1]
                joined[-1][2].extend(run[2])
            else:
                joined.append(run)
        runs = joined
    return runs


def diarize(audio_path: str, segments):
    """
    Label transcript segments with speakers.

    segments are faster-whisper segments. Word timestamps let speaker changes
    inside a segment be found; such a segment is split into parts at word
    boundaries, and the parts have empty token lists.

    Returns a list of (segment, speaker_label) pairs in time order.
    """
    if _extractor is None:
        raise RuntimeError("Diarizer not loaded. Call diarizer.load() first.")
    if not segments:
        return []

    audio = _load_audio(audio_path, target_sr=_SAMPLE_RATE)
    chunks = _split_into_chunks(segments)
    embeddings = [_embed(audio, c.start, c.end) for c in chunks]
    durations = np.array([max(c.end - c.start, 0.0) for c in chunks])
    usable = np.array([e is not None for e in embeddings])

    labels = np.full(len(chunks), -1)
    reliable = usable & (durations >= _MIN_EMBED_SECONDS)
    if reliable.sum() < 2:
        reliable = usable
    if reliable.any():
        rows = np.flatnonzero(reliable)
        matrix = np.array([embeddings[r] for r in rows])
        labels[rows] = _cluster(matrix, durations[rows], _num_speakers, _threshold)
        # Short chunks have noisier embeddings; give them the closest speaker.
        centroids = np.array([
            (matrix[labels[rows] == k] * durations[rows][labels[rows] == k, None]).sum(0)
            for k in range(labels[rows].max() + 1)
        ])
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
        for r in np.flatnonzero(usable & ~reliable):
            labels[r] = int(np.argmax(centroids @ embeddings[r]))
        # Chunks too short to embed take the label of the nearest labeled chunk.
        labeled = np.flatnonzero(labels >= 0)
        for r in np.flatnonzero(labels < 0):
            labels[r] = labels[labeled[np.argmin(np.abs(labeled - r))]]
    else:
        labels[:] = 0

    by_segment = collections.defaultdict(list)
    for chunk, label in zip(chunks, labels):
        by_segment[chunk.segment].append((label, chunk))
    segment_runs = [
        _speaker_runs([label for label, _ in by_segment[index]], [chunk for _, chunk in by_segment[index]])
        for index in range(len(segments))
    ]
    # Number speakers in order of first appearance, after smoothing.
    order = {}
    for runs in segment_runs:
        for label, _, _ in runs:
            order.setdefault(label, len(order))

    result = []
    for seg, runs in zip(segments, segment_runs):
        if len(runs) == 1:
            result.append((seg, f"SPEAKER_{order[runs[0][0]]:02d}"))
            continue
        for position, (label, _, words) in enumerate(runs):
            part = dataclasses.replace(
                seg,
                start=seg.start if position == 0 else words[0].start,
                end=seg.end if position == len(runs) - 1 else words[-1].end,
                text="".join(w.word for w in words),
                tokens=[],
                words=words,
            )
            result.append((part, f"SPEAKER_{order[label]:02d}"))
    return result
