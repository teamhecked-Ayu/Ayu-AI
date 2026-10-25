import contextlib
import io
import os
import tempfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ayu.cli import main
from ayu.settings import Settings
from ayu.provider import LlamaCppProvider, ProviderError
from ayu.store import Store


class BasicModeTests(unittest.TestCase):
    def test_local_commands_work_without_api_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            env={"AYU_DATABASE":str(Path(tmp)/"db.sqlite3"),"AYU_MODE":"basic"}
            out=io.StringIO()
            with patch.dict(os.environ,env,clear=True),contextlib.redirect_stdout(out):
                self.assertEqual(main(["memory","add","--category","preference","short answers"]),0)
                self.assertEqual(main(["memory","search","short"]),0)
                self.assertEqual(main(["status"]),0)
            self.assertIn("short answers",out.getvalue())
            self.assertIn("API key: not configured",out.getvalue())

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


class LocalProviderTests(unittest.TestCase):
    def settings(self,model,exe):
        return Settings(api_key=None,base_url="https://unused.invalid/v1",model=None,database_path=Path("/tmp/ayu-test.db"),local_model_path=model,llama_cli=exe)

    def test_missing_local_model_is_clear_and_no_download(self):
        provider=LlamaCppProvider(self.settings(None,"llama-cli"))
        with self.assertRaisesRegex(ProviderError,"never downloads"):
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


if __name__=="__main__": unittest.main()
