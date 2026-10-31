# Task 3 live experiment: CPU asset preparation

This stage verifies/downloads assets only. It does not instantiate a language model or run a forward pass. Leave Colab on CPU.

Use the original Qwen/Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` and original DigitalLearningGmbH/MATH-lighteval revision `0530c78699ea5e8eb5530600900e1f328b48acad`. The script checks all 1000 supplied original MATH row IDs against the README's default_rng(42) sample and checks the 50 saved live problems/references against the pinned original dataset. It does not change the saved default_rng(6304) development/evaluation split.

The tokenizer constructs native system/user chat prompts using the exact manual instruction. Base prompts, IDs and per-call seed rules are recorded before GPU allocation. The generation-input file excludes gold answers. References are kept separately for later CPU grading. The model files are downloaded from the pinned public repository and hashed, comparing published LFS hashes where available. Weights/config/tokenizer remain under `task3/assets/live_model/<revision>` and must stay outside git.

Each question will need five physical generation calls: one 3072-token single response, three independent 1024-token responses, and one 2048-token critique/revision response. The first parallel response is reused as the revision's initial assistant response. Each compared strategy has a 3072-token allowance, but actual tokens/runtime will be measured; the shared initial's cost is charged to both strategies. Across 50 questions the physical maximum is 409600 output tokens, with 250 independent call seeds. The frozen recipe prescribes FP16, no quantization, batch 1, temperature .7, top-p .8, top-k 20 and repetition penalty 1. No generation takes place during asset preparation.

The later GPU runner must check these assets and package versions, save each completed call for resuming, perform a separate smoke/warmup, and release the runtime after the complete queue. Report complete generated tokens and retained tokens excluding final EOS separately; critique/revision prompt and critique output count toward sequential cost. Generation is never selected or stopped using gold feedback. Live scoring and final plots/analysis stay on CPU after the GPU queue.

Commands:

```bash
python -m unittest discover -s code -p test_task3_live_cpu.py -v
python code/task3_live_assets.py --task-dir /content/drive/MyDrive/ATML/PA3/task3
```
