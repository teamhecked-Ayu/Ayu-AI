"""AyuLM-native byte-level decoder Transformer and optional training utilities.

This is an educational, small-scale train-from-scratch architecture. It is not a
pretrained assistant model and makes no frontier-quality claims.
"""
from __future__ import annotations
import hashlib, json, math, mmap, os, tempfile, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from importlib import import_module

ARCHITECTURE = "ayulm_native_decoder"
ARCH_VERSION = 1
TOKENIZER_ID = "ayulm_utf8_byte_v1"
SPECIAL = {"pad": 0, "bos": 1, "eos": 2, "unk": 3}
VOCAB_SIZE = 260

class ByteTokenizer:
    """Reversible UTF-8 byte tokenizer; no learned vocabulary/training needed."""
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [b + 4 for b in text.encode("utf-8")]
        return ([SPECIAL["bos"]] if add_bos else []) + ids + ([SPECIAL["eos"]] if add_eos else [])
    def decode(self, ids: list[int]) -> str:
        raw = bytes(i - 4 for i in ids if 4 <= int(i) < VOCAB_SIZE)
        return raw.decode("utf-8", errors="replace")
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix="ayu-tokenizer-",suffix=".tmp",dir=path.parent)
        try:
            with os.fdopen(fd,"w",encoding="utf-8") as stream: json.dump({"id":TOKENIZER_ID,"vocab_size":VOCAB_SIZE,"special_tokens":SPECIAL},stream,indent=2); stream.write("\n")
            os.chmod(tmp,0o600); Path(tmp).replace(path)
        except BaseException:
            try: os.unlink(tmp)
            except OSError: pass
            raise

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    seq_len: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.0
    def validate(self) -> None:
        if self.vocab_size != VOCAB_SIZE: raise ValueError("native tokenizer currently requires vocab_size=260")
        if self.seq_len < 2 or self.seq_len>8192 or self.d_model < 8 or self.d_model>4096 or self.n_layers < 1 or self.n_layers>64 or self.n_heads < 1 or self.n_heads>64: raise ValueError("model dimensions outside supported safety bounds (seq_len<=8192, d_model<=4096, layers<=64, heads<=64)")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2: raise ValueError("d_model/n_heads must be an even integer for RoPE")
        if not 0 <= self.dropout < 1: raise ValueError("dropout must be in [0,1)")
    def to_dict(self) -> dict[str,Any]: return asdict(self)


def _torch():
    try: import torch
    except ImportError as exc: raise RuntimeError("AyuLM training/inference needs optional PyTorch. Install it with `python -m pip install -e '.[training]'`; core CLI commands stay dependency-free.") from exc
    return torch


def build_model(config: ModelConfig):
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F
    config.validate()
    class CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__(); self.heads=config.n_heads; self.head_dim=config.d_model//config.n_heads
            self.qkv=nn.Linear(config.d_model,3*config.d_model,bias=False); self.proj=nn.Linear(config.d_model,config.d_model,bias=False); self.dropout=nn.Dropout(config.dropout)
        def forward(self,x):
            b,t,c=x.shape; q,k,v=self.qkv(x).split(c,dim=-1)
            q=q.view(b,t,self.heads,self.head_dim).transpose(1,2); k=k.view(b,t,self.heads,self.head_dim).transpose(1,2); v=v.view(b,t,self.heads,self.head_dim).transpose(1,2)
            # Rotary position embedding, computed in float for numerical stability.
            inv=1.0/(10000.0**(torch.arange(0,self.head_dim,2,device=x.device,dtype=torch.float32)/self.head_dim))
            pos=torch.arange(t,device=x.device,dtype=torch.float32); phase=torch.outer(pos,inv)
            cos=phase.cos().to(q.dtype)[None,None,:,:]; sin=phase.sin().to(q.dtype)[None,None,:,:]
            def rotate(a):
                a1,a2=a[...,0::2],a[...,1::2]
                return torch.stack((a1*cos-a2*sin,a1*sin+a2*cos),dim=-1).flatten(-2)
            q,k=rotate(q),rotate(k)
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=self.dropout.p if self.training else 0.0,is_causal=True)
            return self.dropout(self.proj(y.transpose(1,2).contiguous().view(b,t,c)))
    class Block(nn.Module):
        def __init__(self):
            super().__init__(); self.norm1=nn.LayerNorm(config.d_model); self.attn=CausalSelfAttention(); self.norm2=nn.LayerNorm(config.d_model); self.ff=nn.Sequential(nn.Linear(config.d_model,4*config.d_model),nn.GELU(),nn.Linear(4*config.d_model,config.d_model),nn.Dropout(config.dropout))
        def forward(self,x): x=x+self.attn(self.norm1(x)); return x+self.ff(self.norm2(x))
    class AyuLM(nn.Module):
        def __init__(self):
            super().__init__(); self.config=config; self.token_embedding=nn.Embedding(config.vocab_size,config.d_model); self.drop=nn.Dropout(config.dropout); self.blocks=nn.ModuleList([Block() for _ in range(config.n_layers)]); self.norm=nn.LayerNorm(config.d_model); self.output=nn.Linear(config.d_model,config.vocab_size,bias=False); self.output.weight=self.token_embedding.weight
            self.apply(self._init_weights)
        @staticmethod
        def _init_weights(module):
            if isinstance(module,nn.Linear): nn.init.normal_(module.weight,mean=0.0,std=0.02)
            elif isinstance(module,nn.Embedding): nn.init.normal_(module.weight,mean=0.0,std=0.02)
        def forward(self,idx,targets=None):
            if idx.ndim!=2 or idx.shape[1]>config.seq_len: raise ValueError("input must have shape [batch, sequence] within configured seq_len")
            x=self.drop(self.token_embedding(idx))
            for block in self.blocks: x=block(x)
            logits=self.output(self.norm(x)); loss=F.cross_entropy(logits.reshape(-1,config.vocab_size),targets.reshape(-1)) if targets is not None else None
            return logits,loss
        @torch.no_grad()
        def generate(self,idx,max_new_tokens=64,temperature=1.0,top_k=None):
            if temperature<=0: raise ValueError("temperature must be positive")
            self.eval()
            for _ in range(max_new_tokens):
                logits,_=self(idx[:,-config.seq_len:]); logits=logits[:,-1,:]/temperature
                if top_k is not None:
                    k=max(1,min(int(top_k),logits.shape[-1])); threshold=torch.topk(logits,k).values[:,-1,None]; logits=logits.masked_fill(logits<threshold,-float("inf"))
                probs=torch.softmax(logits,dim=-1); nxt=torch.multinomial(probs,num_samples=1); idx=torch.cat((idx,nxt),dim=1)
            return idx
    return AyuLM()


def parameter_count(model) -> int: return sum(p.numel() for p in model.parameters())

def _perplexity(loss: float) -> float | None:
    if not math.isfinite(loss) or loss>math.log(float.fromhex("0x1.fffffffffffffp+1023")): return None
    return math.exp(loss)

def _sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def load_checkpoint(path: Path, device: str = "cpu", include_optimizer: bool = False):
    torch=_torch()
    if device!="cpu" and not device.startswith("cuda"): raise RuntimeError("AyuLM currently supports device='cpu' or an available CUDA device such as 'cuda'/'cuda:0'.")
    if device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is unavailable in this PyTorch installation; use --device cpu.")
    try: data=torch.load(path,map_location=device,weights_only=True)
    except TypeError: raise RuntimeError("Installed PyTorch lacks safe weights_only checkpoint loading; upgrade to PyTorch >=2.1.")
    except Exception as exc: raise RuntimeError(f"Could not safely load checkpoint: {exc}") from exc
    if not isinstance(data,dict) or data.get("architecture")!=ARCHITECTURE or data.get("architecture_version")!=ARCH_VERSION or data.get("tokenizer_id")!=TOKENIZER_ID:
        raise RuntimeError("Unsupported or incompatible AyuLM checkpoint metadata.")
    config=ModelConfig(**data["model_config"]); config.validate(); model=build_model(config)
    try: model.load_state_dict(data["model_state"],strict=True)
    except Exception as exc: raise RuntimeError(f"Checkpoint weights do not match declared model configuration: {exc}") from exc
    data.pop("model_state",None)
    if not include_optimizer: data.pop("optimizer_state",None)
    model.to(device); model.eval(); return model,data


def _write_registry(checkpoint: Path, config: ModelConfig, count: int, data_path: Path, validation: dict, registry: Path, training_config: dict) -> None:
    registry.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    entries=[]
    if registry.exists():
        try: entries=json.loads(registry.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError): entries=[]
    if not isinstance(entries,list): entries=[]
    entries=[entry for entry in entries if isinstance(entry,dict)]
    digest=_sha256_file(data_path)
    record={"model_id":checkpoint.stem,"architecture":ARCHITECTURE,"architecture_version":ARCH_VERSION,"parameter_count":count,"tokenizer_id":TOKENIZER_ID,"checkpoint_path":str(checkpoint.resolve()),"model_config":config.to_dict(),"training_config":training_config,"training_data":{"path":str(data_path.resolve()),"size_bytes":data_path.stat().st_size,"sha256":digest,"provenance":"user-provided corpus; license not validated by software"},"license":"user must verify data/model license","created_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"validation":validation,"backend":"PyTorch","context_length":config.seq_len}
    entries=[e for e in entries if e.get("model_id")!=checkpoint.stem]+[record]
    fd,tmp=tempfile.mkstemp(prefix="ayu-registry-",suffix=".tmp",dir=registry.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as stream: json.dump(entries,stream,indent=2); stream.write("\n")
        os.chmod(tmp,0o600); Path(tmp).replace(registry)
    except BaseException:
        try: os.unlink(tmp)
        except OSError: pass
        raise


def train(data_path: Path, output: Path, steps: int, config: ModelConfig, seed: int=42, resume: Path|None=None, registry: Path|None=None, eval_interval: int=20, batch_size: int=8, learning_rate: float=3e-4, device: str="cpu", progress: Callable[[dict],None]|None=None) -> dict:
    torch=_torch(); config.validate()
    if steps<1 or eval_interval<1 or batch_size<1 or learning_rate<=0: raise ValueError("steps, eval_interval, batch_size, and learning_rate must be positive")
    data_path=data_path.expanduser().resolve()
    if not data_path.is_file(): raise FileNotFoundError(f"Training corpus not found: {data_path}")
    minimum=2*(config.seq_len+2)
    if data_path.stat().st_size<minimum: raise ValueError(f"Corpus has {data_path.stat().st_size} bytes; need at least {minimum} bytes for disjoint train/validation sequences at seq_len={config.seq_len}.")
    with data_path.open("rb") as source:
        with mmap.mmap(source.fileno(),0,access=mmap.ACCESS_READ) as corpus:
            return _train_mapped(corpus,data_path,output,steps,config,seed,resume,registry,eval_interval,batch_size,learning_rate,device,torch,progress)


def _train_mapped(corpus, data_path: Path, output: Path, steps: int, config: ModelConfig, seed: int, resume: Path|None, registry: Path|None, eval_interval: int, batch_size: int, learning_rate: float, device: str, torch, progress: Callable[[dict],None]|None) -> dict:
    if device!="cpu" and not device.startswith("cuda"): raise ValueError("AyuLM supports only CPU or CUDA devices.")
    if device.startswith("cuda") and not torch.cuda.is_available(): raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
    corpus_len=len(corpus); min_part=config.seq_len+2
    split=min(max(min_part,int(corpus_len*0.9)),corpus_len-min_part)
    train_range=(0,split); val_range=(split,corpus_len)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and device.startswith("cuda"): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    start=0; dataset_digest=_sha256_file(data_path)
    if resume:
        model,checkpoint=load_checkpoint(resume,device,include_optimizer=True)
        if checkpoint["model_config"]!=config.to_dict(): raise ValueError("Resume checkpoint configuration does not match requested model configuration.")
        if checkpoint.get("training_data_sha256")!=dataset_digest: raise ValueError("Training corpus contents differ from the checkpoint. Use the original corpus to resume; start a new run to train on different data.")
        training_config=checkpoint.get("training_config",{})
        batch_size=int(training_config.get("batch_size",batch_size)); learning_rate=float(training_config.get("learning_rate",learning_rate))
        optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
        if "optimizer_state" not in checkpoint: raise ValueError("Checkpoint has no optimizer state and cannot resume training.")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "torch_rng_state" in checkpoint: torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        stored_cuda=bool(checkpoint.get("cuda_rng_states")); current_cuda=str(device).startswith("cuda")
        if stored_cuda!=current_cuda: raise ValueError("Checkpoint CUDA/CPU mode differs; resume using the original device type.")
        if current_cuda and stored_cuda:
            if len(checkpoint["cuda_rng_states"])!=torch.cuda.device_count(): raise ValueError("CUDA device count differs from checkpoint; resume on the original hardware configuration.")
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        start=int(checkpoint.get("step",0))
    else:
        model=build_model(config).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    output=output.expanduser().resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    tokenizer=ByteTokenizer()
    def batch(source_range):
        lo,hi=source_range; high=(hi-lo)-config.seq_len
        if high<1: raise ValueError("Corpus split is too small for configured sequence length")
        offsets=torch.randint(0,high,(batch_size,)).tolist(); xs=[]; ys=[]
        for offset in offsets:
            pos=lo+offset; xs.append([b+4 for b in corpus[pos:pos+config.seq_len]]); ys.append([b+4 for b in corpus[pos+1:pos+config.seq_len+1]])
        return torch.tensor(xs,dtype=torch.long,device=device),torch.tensor(ys,dtype=torch.long,device=device)
    def evaluate():
        model.eval(); results={}
        with torch.no_grad():
            for name,source in (("train",train_range),("validation",val_range)):
                x,y=batch(source); _,loss=model(x,y)
                if not torch.isfinite(loss): raise RuntimeError("non-finite loss during validation; stop and check corpus/configuration")
                value=float(loss.item()); results[name+"_loss"]=value; results[name+"_perplexity"]=_perplexity(value)
        model.train(); return results
    metrics={}
    for step in range(start+1,start+steps+1):
        model.train(); x,y=batch(train_range); optimizer.zero_grad(set_to_none=True); _,loss=model(x,y)
        if not torch.isfinite(loss): raise RuntimeError("non-finite training loss; stop and check corpus/configuration")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        if step==start+1 or step%eval_interval==0 or step==start+steps:
            metrics=evaluate(); training_config={"seed":seed,"batch_size":batch_size,"learning_rate":learning_rate,"eval_interval":eval_interval,"target_steps":step}
            saved={"architecture":ARCHITECTURE,"architecture_version":ARCH_VERSION,"model_config":config.to_dict(),"tokenizer_id":TOKENIZER_ID,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"torch_rng_state":torch.get_rng_state(),"cuda_rng_states":torch.cuda.get_rng_state_all() if torch.cuda.is_available() and str(device).startswith("cuda") else [],"step":step,"seed":seed,"training_config":training_config,"metrics":metrics,"training_data_path":str(data_path),"training_data_sha256":dataset_digest}
            fd,tmp=tempfile.mkstemp(prefix="ayu-checkpoint-",suffix=".tmp",dir=output.parent); os.close(fd)
            try:
                torch.save(saved,tmp); os.chmod(tmp,0o600); Path(tmp).replace(output); tokenizer.save(output.with_suffix(".tokenizer.json"))
            except BaseException:
                try: os.unlink(tmp)
                except OSError: pass
                raise
            if progress: progress({"step":step,**metrics})
    count=parameter_count(model)
    if registry is None: registry=Path.home()/".ayu-ai"/"model-registry.json"
    _write_registry(output,config,count,data_path,metrics,registry,{"seed":seed,"batch_size":batch_size,"learning_rate":learning_rate,"eval_interval":eval_interval,"additional_steps":steps})
    return {"checkpoint":str(output),"parameters":count,"step":start+steps,**metrics}


_LEGACY_EXPORTS={"AyuLMConfig","AyuLMTokenizer","AyuLMModel","TinyDecoderBlock"}

def __getattr__(name: str):
    """Lazy compatibility bridge for the original NumPy prototype API."""
    if name in _LEGACY_EXPORTS:
        legacy=import_module(".legacy_model",__package__)
        return getattr(legacy,name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
