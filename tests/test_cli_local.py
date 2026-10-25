import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ayu.cli import main
from ayu.settings import Settings
from ayu.provider import LlamaCppProvider, ProviderError
from ayu.store import Store


class BasicModeTests(unittest.TestCase):
    def test_self_test_is_offline_and_does_not_load_user_configuration(self):
        output = io.StringIO()
        with patch("ayu.settings.Settings.load", side_effect=AssertionError("must not load user settings")):
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["test"]), 0)
        self.assertIn('"status": "passed"', output.getvalue())
        self.assertIn('"sqlite_chat"', output.getvalue())

    def test_local_commands_work_without_api_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            env={"AYU_DATABASE":str(Path(tmp)/"db.sqlite3"),"AYU_MODE":"basic"}
            out=io.StringIO()
            with patch.dict(os.environ,env,clear=True),contextlib.redirect_stdout(out):
                self.assertEqual(main(["memory","add","--category","preference","short answers"]),0)
                self.assertEqual(main(["memory","search","short"]),0)
                self.assertEqual(main(["status"]),0)
            self.assertIn("short answers",out.getvalue())
            self.assertIn("Selected AI credential: not configured",out.getvalue())

    def test_basic_mode_is_not_pretended_to_be_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ,{"AYU_DATABASE":str(Path(tmp)/"db.sqlite3"),"AYU_MODE":"basic"},clear=True):
                err=io.StringIO()
                with contextlib.redirect_stderr(err): self.assertEqual(main(["chat","--once","hello"]),2)
            self.assertIn("does not perform LLM inference",err.getvalue())

    def test_projects_and_tasks_are_persistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Store(Path(tmp)/"db.sqlite3")
            project=db.create_project("Sample")
            task=db.add_task(project,"verify build")
            self.assertEqual(db.get_project(project)["name"],"Sample")
            self.assertEqual(db.list_tasks(project)[0]["title"],"verify build")
            self.assertTrue(db.update_task_status(task,"completed"))
            self.assertEqual(db.list_tasks(project)[0]["status"],"completed")

    def test_memory_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            env={"AYU_DATABASE":str(Path(tmp)/"db.sqlite3"),"AYU_MODE":"basic","AYU_MEMORY_ENABLED":"false"}
            err=io.StringIO()
            with patch.dict(os.environ,env,clear=True),contextlib.redirect_stderr(err):
                self.assertEqual(main(["memory","add","not saved"]),2)
            self.assertIn("disabled",err.getvalue())

    def test_project_state_updates_are_structured(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Store(Path(tmp)/"db.sqlite3")
            project=db.create_project("continuity")
            state={"goals":["preserve verified progress"],"current_state":"tests running"}
            self.assertTrue(db.set_project_state(project,state))
            self.assertIn("preserve verified progress",db.get_project(project)["state_json"])

    def test_task_cli_enforces_prerequisites_and_reports_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            env={"AYU_DATABASE":str(Path(tmp)/"db.sqlite3"),"AYU_MODE":"basic"}
            output=io.StringIO(); errors=io.StringIO()
            with patch.dict(os.environ,env,clear=True),contextlib.redirect_stdout(output),contextlib.redirect_stderr(errors):
                self.assertEqual(main(["project","create","ordered"]),0)
                self.assertEqual(main(["tasks","add","ordered","prerequisite"]),0)
                self.assertEqual(main(["tasks","add","ordered","follow-up","--depends-on","1"]),0)
                self.assertEqual(main(["tasks","inspect","2"]),0)
                before=output.getvalue()
                self.assertIn('"ready": false',before)
                self.assertEqual(main(["tasks","complete","2"]),2)
                self.assertIn("blocked by incomplete prerequisites",errors.getvalue())
                self.assertEqual(main(["tasks","start","1"]),0)
                self.assertEqual(main(["tasks","complete","1"]),0)
                self.assertEqual(main(["tasks","start","2"]),0)
                self.assertEqual(main(["tasks","complete","2"]),0)


class LocalProviderTests(unittest.TestCase):
    def settings(self,model,exe):
        return Settings(api_key=None,base_url="https://unused.invalid/v1",model=None,database_path=Path("/tmp/ayu-test.db"),local_model_path=model,llama_cli=exe)

    def test_missing_local_model_is_clear_and_no_download(self):
        provider=LlamaCppProvider(self.settings(None,"llama-cli"))
        with self.assertRaisesRegex(ProviderError,"never downloads"):
            provider.validate()

    def test_empty_gguf_is_rejected_before_runtime_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            model=Path(temporary)/"empty.gguf"; model.write_bytes(b"")
            provider=LlamaCppProvider(self.settings(model,"missing-llama-cli"))
            with self.assertRaisesRegex(ProviderError,"empty"):
                provider.validate()

    @unittest.skipIf(os.name=="nt","fake executable uses POSIX permissions")
    def test_local_inference_uses_prompt_file_and_sanitized_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); model=root/"model.gguf"; model.write_bytes(b"test-model-placeholder")
            script=root/"llama-cli"; script.write_text("#!/usr/bin/env python3\nimport os,sys\nargs=sys.argv[1:]\nassert 'AYU_TEST_SECRET' not in os.environ\np=args[args.index('-f')+1]\nassert 'private prompt' in open(p,encoding='utf-8').read()\nprint('local answer')\n",encoding="utf-8"); script.chmod(0o700)
            provider=LlamaCppProvider(self.settings(model,str(script)))
            created=[]; original=tempfile.NamedTemporaryFile
            def tracked_tempfile(*args,**kwargs):
                item=original(*args,**kwargs); created.append(Path(item.name)); return item
            with patch.dict(os.environ,{"AYU_TEST_SECRET":"must-not-leak"}):
                with patch("ayu.provider.tempfile.NamedTemporaryFile",side_effect=tracked_tempfile):
                    answer=provider.complete([{"role":"user","content":"private prompt"}])
            self.assertEqual(answer,"local answer")
            self.assertEqual(len(created),1)
            self.assertFalse(created[0].exists())

    @unittest.skipIf(os.name=="nt","fake executable uses POSIX permissions")
    def test_local_runtime_output_is_bounded_and_prompt_file_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); model=root/"model.gguf"; model.write_bytes(b"model")
            script=root/"llama-cli"; script.write_text("#!/usr/bin/env python3\nimport sys\nsys.stdout.write('x' * (5*1024*1024))\n",encoding="utf-8"); script.chmod(0o700)
            created=[]; original=tempfile.NamedTemporaryFile
            def tracked(*args,**kwargs):
                stream=original(*args,**kwargs); created.append(Path(stream.name)); return stream
            provider=LlamaCppProvider(self.settings(model,str(script)))
            with patch("ayu.provider.tempfile.NamedTemporaryFile",side_effect=tracked):
                with self.assertRaisesRegex(ProviderError,"4 MiB"):
                    provider.complete([{"role":"user","content":"sensitive prompt"}])
            self.assertEqual(len(created),1); self.assertFalse(created[0].exists())

    @unittest.skipIf(os.name=="nt","fake executable uses POSIX permissions")
    def test_local_runtime_timeout_and_stderr_suppression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); model=root/"model.gguf"; model.write_bytes(b"model")
            script=root/"llama-cli"; script.write_text("#!/usr/bin/env python3\nimport sys\nsys.stderr.write('sensitive prompt echo')\nsys.exit(7)\n",encoding="utf-8"); script.chmod(0o700)
            provider=LlamaCppProvider(self.settings(model,str(script)))
            with self.assertRaises(ProviderError) as caught:
                provider.complete([{"role":"user","content":"sensitive prompt"}])
            self.assertNotIn("sensitive prompt echo",str(caught.exception))
            script.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(3)\n",encoding="utf-8"); script.chmod(0o700)
            limited=Settings(api_key=None,base_url="https://unused.invalid/v1",model=None,database_path=Path("/tmp/ayu-test.db"),local_model_path=model,llama_cli=str(script),local_timeout=0.1)
            with self.assertRaisesRegex(ProviderError,"exceeded 0.1 seconds"):
                LlamaCppProvider(limited).complete([{"role":"user","content":"prompt"}])


if __name__=="__main__": unittest.main()
