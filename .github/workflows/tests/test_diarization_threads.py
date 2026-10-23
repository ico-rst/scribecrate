"""Diarization regressions without ONNX or downloads; speaker tests need NumPy."""
import asyncio
import dataclasses
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from test_api_validation import api_server

try:
    import numpy
except ImportError:  # the speaker tests are skipped without NumPy
    numpy = None


OPTION = "WHISPER_DIARIZE_THREADS"
BAD = [
    "0", "00", "01", "-1", "+1", "1.5", "1_000", "2147483648",
    "9" * 10000, "\u0666", "\uff16", "6\u00a0", "\u00a06", "6\n7",
    "6\ninvalid", "6 7", "'\"6\"'", "'6", '6"', "NaN", "$(id)",
]
VALID = [("", 2), ("1", 1), ("6", 6), ("2147483647", 2147483647),
         (" \t\r\n\v\f' 6 '\t", 6), ('""', 2), ("' \r\n '", 2)]
ROOT = Path(__file__).resolve().parents[3]


class DiarizationOptionTests(unittest.TestCase):
    def setUp(self):
        for patch in (
            mock.patch.dict(os.environ, {
                OPTION: "", "WHISPER_THREADS": "", "WHISPER_DIARIZATION": "true",
                "WHISPER_CONDITION_ON_PREVIOUS_TEXT": "", "WHISPER_NO_REPEAT_NGRAM_SIZE": "",
                "WHISPER_DEVICE": "",
            }),
            mock.patch.object(api_server, "_diarize_threads", 2),
            mock.patch.object(api_server, "_diarization_enabled", False),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        for name in ("_model", "_model_name", "_beam_size", "_max_request_beam",
                     "_word_timestamps", "_max_upload_bytes"):
            patch = mock.patch.object(api_server, name, getattr(api_server, name))
            patch.start()
            self.addCleanup(patch.stop)

    def test_defaults_inheritance_and_normalization(self):
        for value, expected in VALID:
            with self.subTest(value=value):
                os.environ[OPTION] = value
                api_server._load_diarization_options()
                self.assertEqual(api_server._diarize_threads, expected)
        os.environ.pop(OPTION)
        os.environ.pop("WHISPER_THREADS", None)
        api_server._load_diarization_options()
        self.assertEqual(api_server._diarize_threads, 2)
        os.environ["WHISPER_THREADS"] = "4"
        api_server._load_diarization_options()
        self.assertEqual(api_server._diarize_threads, 4)
        os.environ[OPTION] = "1"
        api_server._load_diarization_options()
        self.assertEqual(api_server._diarize_threads, 1)

    def test_invalid_override_fails_in_both_modes_without_changing_state(self):
        for enabled in ("true", "false"):
            os.environ["WHISPER_DIARIZATION"] = enabled
            for value in BAD:
                with self.subTest(enabled=enabled, value=value[:60]):
                    os.environ[OPTION] = value
                    api_server._diarize_threads = 6
                    with self.assertRaisesRegex(ValueError, OPTION):
                        api_server._load_diarization_options()
                    self.assertEqual(api_server._diarize_threads, 6)

    def test_invalid_startup_never_loads_either_model(self):
        async def start():
            async with api_server._lifespan(api_server.app):
                self.fail("Invalid config began serving")
        for enabled in ("true", "false"):
            os.environ.update({OPTION: "0", "WHISPER_DIARIZATION": enabled})
            with mock.patch.object(api_server, "_load_model") as whisper:
                with mock.patch.object(api_server, "_load_diarizer") as diarizer:
                    with self.assertRaisesRegex(ValueError, OPTION):
                        asyncio.run(start())
                    whisper.assert_not_called()
                    diarizer.assert_not_called()

    def test_direct_python_legacy_nonpositive_whisper_is_preserved(self):
        for value in ("0", "-2"):
            os.environ["WHISPER_THREADS"] = value
            with self.assertLogs(api_server.logger, level="WARNING") as logs:
                api_server._load_diarization_options()
            self.assertEqual(api_server._diarize_threads, 1)
            self.assertIn("legacy diarization", " ".join(logs.output))
            whisper = mock.Mock()
            with mock.patch.dict(sys.modules, {"faster_whisper": types.SimpleNamespace(WhisperModel=whisper)}):
                with mock.patch.object(api_server, "_model"), mock.patch.object(api_server, "_model_name"):
                    api_server._load_model()
            self.assertEqual(whisper.call_args.kwargs["cpu_threads"], int(value))

    def test_inherited_overflow_only_fails_when_used(self):
        for value in ("2147483648", "9" * 100):
            os.environ.update({"WHISPER_THREADS": value, "WHISPER_DIARIZATION": "true"})
            with self.assertRaisesRegex(ValueError, "Inherited WHISPER_THREADS"):
                api_server._load_diarization_options()
            os.environ["WHISPER_DIARIZATION"] = "false"
            api_server._load_diarization_options()
        os.environ.update({OPTION: "6", "WHISPER_DIARIZATION": "true"})
        api_server._load_diarization_options()
        self.assertEqual(api_server._diarize_threads, 6)

    def test_enabled_load_propagates_exact_config_and_retains_failure_policy(self):
        fake = types.ModuleType("diarizer")
        fake.load = mock.Mock()
        os.environ.update({OPTION: "6", "WHISPER_DIARIZE_NUM_SPEAKERS": "2",
                           "WHISPER_DIARIZE_THRESHOLD": "0.7", "HF_HOME": "/fixture/cache"})
        with mock.patch.dict(sys.modules, {"diarizer": fake}):
            api_server._load_diarization_options()
            with self.assertLogs(api_server.logger, level="INFO") as logs:
                api_server._load_diarizer()
            fake.load.assert_called_once_with(cache_dir="/fixture/cache", num_speakers=2,
                                              cluster_threshold=0.7, num_threads=6, provider="cpu")
            self.assertIn("num_threads=6", " ".join(logs.output))
            self.assertTrue(api_server._diarization_enabled)
            fake.load.side_effect = RuntimeError("fixture native load failure")
            with self.assertLogs(api_server.logger, level="WARNING"):
                api_server._load_diarizer()
            self.assertFalse(api_server._diarization_enabled)

    def test_provider_follows_whisper_device(self):
        fake = types.ModuleType("diarizer")
        fake.load = mock.Mock()
        cases = [("", "cpu"), ("cpu", "cpu"), ("auto", "cpu"), ("cuda", "cuda"), (" CUDA ", "cuda")]
        with mock.patch.dict(sys.modules, {"diarizer": fake}):
            for device, expected in cases:
                with self.subTest(device=device):
                    os.environ["WHISPER_DEVICE"] = device
                    with self.assertLogs(api_server.logger, level="INFO") as logs:
                        api_server._load_diarizer()
                    self.assertEqual(fake.load.call_args.kwargs["provider"], expected)
                    self.assertIn("on " + expected, " ".join(logs.output))
            os.environ.pop("WHISPER_DEVICE")
            api_server._load_diarizer()
            self.assertEqual(fake.load.call_args.kwargs["provider"], "cpu")

    def test_disabled_diarization_has_no_optional_imports(self):
        original = __import__
        def guarded(name, *args, **kwargs):
            if name in ("diarizer", "sherpa_onnx", "numpy"):
                self.fail("Disabled diarization imported " + name)
            return original(name, *args, **kwargs)
        for inherited in ("0", "-2", "2147483648", "6"):
            os.environ.update({"WHISPER_THREADS": inherited, "WHISPER_DIARIZATION": "false"})
            with mock.patch("builtins.__import__", side_effect=guarded):
                api_server._load_diarization_options()
                api_server._load_diarizer()

    def test_startup_resolves_threads_before_whisper_loading(self):
        os.environ[OPTION] = "6"
        async def start():
            async with api_server._lifespan(api_server.app):
                self.assertEqual(api_server._diarize_threads, 6)
        def check():
            self.assertEqual(api_server._diarize_threads, 6)
        with mock.patch.object(api_server, "_load_model", side_effect=check):
            with mock.patch.object(api_server, "_load_diarizer", side_effect=check):
                asyncio.run(start())


class DiarizerConstructionTests(unittest.TestCase):
    def setUp(self):
        self.native = types.ModuleType("sherpa_onnx")
        self.valid = True
        self.native.SpeakerEmbeddingExtractorConfig = mock.Mock(
            side_effect=lambda **kwargs: types.SimpleNamespace(validate=lambda: self.valid, **kwargs))
        self.native.SpeakerEmbeddingExtractor = mock.Mock(return_value=types.SimpleNamespace(dim=192))
        spec = importlib.util.spec_from_file_location("diarizer_thread_fixture", ROOT / "diarizer.py")
        self.diarizer = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"numpy": types.ModuleType("numpy"), "sherpa_onnx": self.native}):
            spec.loader.exec_module(self.diarizer)
        self.model_patch = mock.patch.object(self.diarizer, "_ensure_model", return_value="emb.onnx")
        self.models = self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def test_model_receives_threads_and_legacy_positionals_work(self):
        for threads in (1, 2, 6, 2147483647):
            self.diarizer.load("/cache", 2, 0.7, num_threads=threads)
            self.native.SpeakerEmbeddingExtractorConfig.assert_called_with(model="emb.onnx", num_threads=threads,
                                                                          provider="cpu")
            self.assertEqual((self.diarizer._num_speakers, self.diarizer._threshold), (2, 0.7))
            self.assertTrue(self.diarizer.is_loaded())
        self.diarizer.load("/cache", -1, 0.5)
        self.native.SpeakerEmbeddingExtractorConfig.assert_called_with(model="emb.onnx", num_threads=1,
                                                                      provider="cpu")
        self.assertEqual((self.diarizer._num_speakers, self.diarizer._threshold), (-1, 0.5))
        self.diarizer.load("/cache", 0)
        self.assertEqual((self.diarizer._num_speakers, self.diarizer._threshold), (-1, 0.45))

    def test_direct_invalid_values_fail_before_download_or_native_setup(self):
        for value in (0, -1, 2147483648, 10**100, True, False, 1.0, "6", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "num_threads"):
                    self.diarizer.load(num_threads=value)
        self.models.assert_not_called()
        self.native.SpeakerEmbeddingExtractorConfig.assert_not_called()

    def test_model_receives_provider(self):
        self.diarizer.load("/cache", 2, 0.7, 2, "cuda")
        self.native.SpeakerEmbeddingExtractorConfig.assert_called_with(model="emb.onnx", num_threads=2,
                                                                      provider="cuda")
        for value in ("gpu", "CUDA", "", None, "tensorrt"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "provider"):
                    self.diarizer.load(num_threads=2, provider=value)
        self.models.assert_called_once()

    def test_invalid_native_config_does_not_initialize_extractor(self):
        self.valid = False
        with self.assertRaisesRegex(RuntimeError, "config validation failed"):
            self.diarizer.load(num_threads=2)
        self.native.SpeakerEmbeddingExtractor.assert_not_called()

    def test_cache_reuse_does_not_download(self):
        # Exercise the real resolver against an already cached file.
        self.model_patch.stop()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / self.diarizer._EMB_MODEL_REL).touch()
            with mock.patch.object(self.diarizer, "_download_file") as download:
                self.diarizer.load(directory, num_threads=6)
            download.assert_not_called()

    def test_interrupted_download_leaves_no_model_file(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "model.onnx"
            def failing_curl(args, check):
                Path(args[args.index("-o") + 1]).write_bytes(b"trunc")
                raise subprocess.CalledProcessError(18, args)
            with mock.patch.object(self.diarizer.subprocess, "run", side_effect=failing_curl):
                with self.assertRaises(subprocess.CalledProcessError):
                    self.diarizer._download_file("https://example.invalid/model.onnx", str(dest))
            self.assertFalse(dest.exists())
            def curl(args, check):
                Path(args[args.index("-o") + 1]).write_bytes(b"model")
            with mock.patch.object(self.diarizer.subprocess, "run", side_effect=curl):
                self.diarizer._download_file("https://example.invalid/model.onnx", str(dest))
            self.assertEqual(dest.read_bytes(), b"model")
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["model.onnx"])


@dataclasses.dataclass
class _Word:
    start: float
    end: float
    word: str
    probability: float = 1.0


@dataclasses.dataclass
class _Segment:
    start: float
    end: float
    text: str
    words: list
    tokens: list = dataclasses.field(default_factory=lambda: [50364])


# Each fake voice fills its audio with a distinct sample value; the fake model
# maps that value to a fixed embedding direction.
_VOICES = {1: [1.0, 0.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0, 0.0], 3: [0.0, 0.0, 1.0, 0.0],
           4: [-0.4, -0.6, 0.0, 1.0]}  # voice 4: a stray sound unlike both, nearest to voice 1


class _FakeStream:
    def accept_waveform(self, sample_rate, samples):
        self.samples = samples

    def input_finished(self):
        pass


class _FakeExtractor:
    def create_stream(self):
        return _FakeStream()

    def is_ready(self, stream):
        return len(stream.samples) >= 800

    def compute(self, stream):
        voiced = stream.samples[stream.samples > 0]
        return _VOICES[int(round(float(voiced.mean()) * 10))]


@unittest.skipIf(numpy is None, "NumPy is not installed")
class DiarizerSpeakerTests(unittest.TestCase):
    """Chunking, clustering and segment splitting with a fake embedding model."""

    def setUp(self):
        spec = importlib.util.spec_from_file_location("diarizer_speaker_fixture", ROOT / "diarizer.py")
        self.diarizer = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"sherpa_onnx": types.ModuleType("sherpa_onnx")}):
            spec.loader.exec_module(self.diarizer)
        self.diarizer._extractor = _FakeExtractor()
        self.diarizer._num_speakers = -1
        self.diarizer._threshold = 0.45

    def run_diarizer(self, segments, voiced):
        """voiced: (start, end, voice) spans written into a silent 16 kHz recording."""
        audio = numpy.zeros(int((max(e for _, e, _ in voiced) + 1) * 16000), dtype=numpy.float32)
        for start, end, voice in voiced:
            audio[int(start * 16000):int(end * 16000)] = voice / 10
        with mock.patch.object(self.diarizer, "_load_audio", return_value=audio):
            return self.diarizer.diarize("fixture.wav", segments)

    @staticmethod
    def segment(*words):
        """words: (start, end, text) tuples; returns a segment spanning them."""
        items = [_Word(s, e, " " + t) for s, e, t in words]
        return _Segment(items[0].start, items[-1].end, "".join(w.word for w in items), items)

    def test_alternating_speakers_are_labeled_in_order_of_appearance(self):
        turns = [(0.0, 3.0, 2), (3.5, 6.5, 1), (7.0, 10.0, 2), (10.5, 13.5, 1)]
        segments = [self.segment((s, e, "Hallo zusammen.")) for s, e, _ in turns]
        for speakers in (-1, 2):
            with self.subTest(num_speakers=speakers):
                self.diarizer._num_speakers = speakers
                result = self.run_diarizer(segments, turns)
                self.assertEqual([s for _, s in result], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_01"])
                self.assertEqual([seg for seg, _ in result], segments)

    def test_segment_is_split_where_the_speaker_changes(self):
        segment = self.segment((0.0, 1.0, "Good"), (1.0, 2.5, "morning."), (3.0, 4.0, "Good"), (4.0, 6.0, "morning."))
        voiced = [(0.0, 2.5, 1), (3.0, 6.0, 2), (7.0, 10.0, 1)]
        later = self.segment((7.0, 10.0, "Thanks."))
        result = self.run_diarizer([segment, later], voiced)
        self.assertEqual([s for _, s in result], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])
        first, second, third = (part for part, _ in result)
        self.assertEqual((first.start, first.end, first.text), (0.0, 2.5, " Good morning."))
        self.assertEqual((second.start, second.end, second.text), (3.0, 6.0, " Good morning."))
        self.assertEqual((first.tokens, second.tokens), ([], []))
        self.assertEqual([w.word for w in second.words], [" Good", " morning."])
        self.assertIs(third, later)

    def test_brief_change_inside_a_segment_is_ignored(self):
        segment = self.segment((0.0, 3.0, "Ich"), (3.6, 3.8, "ja"), (4.4, 7.0, "denke."))
        voiced = [(0.0, 3.0, 1), (3.6, 3.8, 2), (4.4, 7.0, 1), (8.0, 11.0, 2)]
        result = self.run_diarizer([segment, self.segment((8.0, 11.0, "Genau."))], voiced)
        self.assertEqual([(seg.text, s) for seg, s in result],
                         [(segment.text, "SPEAKER_00"), (" Genau.", "SPEAKER_01")])

    def test_speaker_numbers_have_no_gaps_after_smoothing(self):
        # Voice 3 appears first, but only as a brief change that is smoothed away.
        segment = self.segment((0.0, 3.0, "Ich"), (3.6, 3.8, "ja"), (4.4, 7.0, "denke."))
        voiced = [(0.0, 3.0, 1), (3.6, 3.8, 3), (4.4, 7.0, 1), (8.0, 11.0, 2), (12.0, 15.0, 3)]
        segments = [segment, self.segment((8.0, 11.0, "Genau.")), self.segment((12.0, 15.0, "Gut."))]
        self.diarizer._num_speakers = 3
        labels = [s for _, s in self.run_diarizer(segments, voiced)]
        self.assertEqual(labels, ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"])

    def test_stray_sound_cannot_take_a_speaker_slot(self):
        turns = [(0.0, 4.0, 1), (4.5, 8.5, 2), (9.0, 10.6, 4), (11.0, 15.0, 1), (15.5, 19.5, 2)]
        segments = [self.segment((s, e, "Text.")) for s, e, _ in turns]
        for speakers in (-1, 2):
            with self.subTest(num_speakers=speakers):
                self.diarizer._num_speakers = speakers
                labels = [s for _, s in self.run_diarizer(segments, turns)]
                self.assertEqual(labels, ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_00", "SPEAKER_01"])

    def test_threshold_and_speaker_count_control_the_number_of_speakers(self):
        turns = [(0.0, 4.0, 1), (4.5, 8.5, 2), (9.0, 13.0, 3)]
        segments = [self.segment((s, e, "Text.")) for s, e, _ in turns]
        cases = [(-1, 0.45, 3), (-1, 2.0, 1), (3, 2.0, 3), (2, 0.45, 2), (1, 0.45, 1)]
        for speakers, threshold, expected in cases:
            with self.subTest(num_speakers=speakers, threshold=threshold):
                self.diarizer._num_speakers, self.diarizer._threshold = speakers, threshold
                labels = {s for _, s in self.run_diarizer(segments, turns)}
                self.assertEqual(len(labels), expected)

    def test_segments_without_words_or_audio_are_labeled(self):
        no_words = _Segment(0.0, 3.0, " Hallo.", None)
        too_short = self.segment((3.2, 3.21, "Ja."))
        voiced = [(0.0, 3.0, 1), (4.0, 7.0, 2)]
        result = self.run_diarizer([no_words, too_short, self.segment((4.0, 7.0, "Okay."))], voiced)
        self.assertEqual([s for _, s in result], ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(self.diarizer.diarize("unused.wav", []), [])


class DiarizedResponseTests(unittest.TestCase):
    """The batch API asks for word timings and outputs the diarizer's segments."""

    def setUp(self):
        from test_api_validation import _FakeInfo, _FakeUpload
        self.upload, self.info = _FakeUpload, _FakeInfo
        fields = dict(seek=0, tokens=[1], temperature=0.0, avg_logprob=-0.1, compression_ratio=1.0,
                      no_speech_prob=0.0)
        whole = types.SimpleNamespace(start=0.0, end=4.0, text=" Hallo. Guten Tag.", words=["w"], **fields)
        self.parts = [types.SimpleNamespace(start=0.0, end=1.5, text=" Hallo.", words=["w"], **fields),
                      types.SimpleNamespace(start=2.0, end=4.0, text=" Guten Tag.", words=["w"], **fields)]
        self.model = mock.Mock()
        self.model.transcribe.side_effect = lambda *a, **k: (iter([whole]), self.info())
        self.fake = types.ModuleType("diarizer")
        self.fake.diarize = mock.Mock(return_value=[(self.parts[0], "SPEAKER_00"), (self.parts[1], "SPEAKER_01")])
        for name, value in (("_model", self.model), ("_model_name", "base"), ("_diarization_enabled", True),
                            ("_max_upload_bytes", 0), ("_word_timestamps", False)):
            patch = mock.patch.object(api_server, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.dict(sys.modules, {"diarizer": self.fake})
        patch.start()
        self.addCleanup(patch.stop)

    def request(self, response_format):
        return asyncio.run(api_server._handle_audio(
            task="transcribe", file=self.upload(), model="whisper-1", language=None, prompt=None,
            response_format=response_format, temperature=0, stream=None, beam=None,
        ))

    def test_verbose_json_uses_split_segments_without_unrequested_words(self):
        body = self.request("verbose_json").content
        self.assertTrue(self.model.transcribe.call_args.kwargs["word_timestamps"])
        self.assertEqual([(s["id"], s["text"], s["speaker"]) for s in body["segments"]],
                         [(0, "Hallo.", "SPEAKER_00"), (1, "Guten Tag.", "SPEAKER_01")])
        self.assertNotIn("words", body)
        self.assertEqual(body["text"], "Hallo. Guten Tag.")

    def test_text_and_subtitles_label_each_speaker(self):
        self.assertEqual(self.request("text").content, "[SPEAKER_00] Hallo.\n[SPEAKER_01] Guten Tag.")
        self.assertIn("[SPEAKER_01] Guten Tag.", self.request("srt").content)
        self.assertIn("[SPEAKER_01] Guten Tag.", self.request("vtt").content)

    def test_failed_diarization_returns_the_transcript_unchanged(self):
        self.fake.diarize.side_effect = RuntimeError("fixture failure")
        with self.assertLogs(api_server.logger, level="WARNING"):
            body = self.request("verbose_json").content
        self.assertEqual([s["text"] for s in body["segments"]], ["Hallo. Guten Tag."])
        self.assertNotIn("speaker", body["segments"][0])


class ShellDiarizationOptionTests(unittest.TestCase):
    def test_shell_python_agreement_and_plain_env_export(self):
        source = (ROOT / "run.sh").read_text()
        helpers = source[source.index("normalize_decoding_option() {"):
                         source.index("# End strict diarization option helpers.")]
        script = ('exiterr() { echo "$1" >&2; exit 1; }\n' + helpers +
                  '\nunset WHISPER_DIARIZE_THREADS\n. "$1"\nconfigure_diarization_threads\n'
                  'exec "$2" -c \'import os; print(repr(os.environ["WHISPER_DIARIZE_THREADS"]))\'\n')
        cases = [(v, "2", e, True) for v, _ in VALID for e in ("true", "false")]
        cases += [(v, "2", e, False) for v in BAD for e in ("true", "false")]
        cases += [("", "6", "true", True), ("", "2147483647", "true", True),
                  ("", "2147483648", "true", False), ("", "2147483648", "false", True),
                  ("1", "2147483648", "true", True)]
        def quote(value):
            return "'" + value.replace("'", "'\"'\"'") + "'"
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "whisper.env"
            for value, inherited, enabled, valid in cases:
                config.write_text(OPTION + "=" + quote(value) + "\nWHISPER_THREADS=" + quote(inherited) +
                                  "\nWHISPER_DIARIZATION=" + quote(enabled) + "\n")
                result = subprocess.run(["bash", "-c", script, "diarization-test", str(config), sys.executable],
                                        text=True, capture_output=True, timeout=10)
                with self.subTest(value=value[:60], inherited=inherited, enabled=enabled):
                    self.assertEqual(result.returncode == 0, valid, result.stderr)
                    with mock.patch.dict(os.environ, {OPTION: value, "WHISPER_THREADS": inherited,
                                                       "WHISPER_DIARIZATION": enabled}):
                        with mock.patch.object(api_server, "_diarize_threads", 2):
                            if valid:
                                api_server._load_diarization_options()
                                self.assertEqual(result.stdout.strip(), repr(api_server._normalized_env_option(OPTION)))
                            else:
                                self.assertIn("WHISPER_", result.stderr)
                                with self.assertRaises(ValueError):
                                    api_server._load_diarization_options()


if __name__ == "__main__":
    unittest.main()
