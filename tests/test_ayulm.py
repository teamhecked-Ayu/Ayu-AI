import importlib.util
import tempfile
import unittest
from pathlib import Path

from ayu.ayulm import ByteTokenizer, ModelConfig


class AyuLMConfigTests(unittest.TestCase):
    def test_invalid_model_configuration_rejected(self):
        with self.assertRaises(ValueError): ModelConfig(seq_len=8,d_model=15,n_heads=2,n_layers=1).validate()


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional AyuLM tests require PyTorch")
class AyuLMTrainingTests(unittest.TestCase):
    def test_train_save_load_evaluate_and_generate_tiny_model(self):
        import torch
        from ayu import ayulm
        torch.set_num_threads(1)
        tokenizer=ByteTokenizer()
        self.assertEqual(tokenizer.decode(tokenizer.encode("Ayu AI ✓")),"Ayu AI ✓")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); corpus=root/"synthetic.txt"
            corpus.write_text("AyuLM synthetic smoke-test corpus. "*80,encoding="utf-8")
            config=ModelConfig(seq_len=8,d_model=16,n_heads=2,n_layers=1,dropout=0.0)
            checkpoint=root/"tiny.pt"; registry=root/"registry.json"
            result=ayulm.train(corpus,checkpoint,steps=1,config=config,seed=7,batch_size=2,eval_interval=1,registry=registry)
            self.assertTrue(checkpoint.is_file()); self.assertGreater(result["parameters"],0)
            model,metadata=ayulm.load_checkpoint(checkpoint)
            self.assertEqual(metadata["architecture"],ayulm.ARCHITECTURE)
            self.assertIn("validation_perplexity",metadata["metrics"])
            prompt=torch.tensor([tokenizer.encode("Ayu")],dtype=torch.long)
            generated=model.generate(prompt,max_new_tokens=2)
            self.assertEqual(generated.shape[1],prompt.shape[1]+2)
            self.assertTrue(registry.is_file())
            resumed=ayulm.train(corpus,checkpoint,steps=1,config=config,seed=7,resume=checkpoint,registry=registry)
            self.assertEqual(resumed["step"],2)
            other=root/"changed.txt"; other.write_text("different training content "*80,encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"corpus contents differ"):
                ayulm.train(other,checkpoint,steps=1,config=config,seed=7,resume=checkpoint,registry=registry)


if __name__=="__main__": unittest.main()
