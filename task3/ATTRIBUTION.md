# Sources and attribution

- Assignment-supplied ATML-PA3-Data and README: original frozen parquet hashes, generation settings, token measurements and field definitions are retained in `configs/supplied_data_inventory.json` and preparation receipts. Raw exports are not redistributed here.
- MATH: DigitalLearningGmbH/MATH-lighteval, revision `0530c78699ea5e8eb5530600900e1f328b48acad`. Original test problems and solution-derived boxed references were checked for the fixed live subset.
- Live model: Qwen/Qwen2.5-3B-Instruct, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Model weights remain external; hashes and model/license file identity are in `configs/live_assets.json`.
- MathVista released metadata and answer normalization: https://github.com/lupantech/MathVista/tree/c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb . Prediction normalization in `code/task3_answers.py` adapts the released `evaluation/calculate_score.py` rules; exact reference decoding is separated to handle literal parenthesized choice text. The source repository's CC BY-SA 4.0 license and original source are retained with the external metadata assets.
- Math-Verify: https://github.com/huggingface/Math-Verify . Parser and verifier are called through their public API; dependencies are pinned.
- Methods: Kang et al., Scalable Best-of-N Selection for Large Language Models via Self-Certainty; Usman et al., VisConf: Quality-Aware Consensus for Best-of-N in Vision-Language Model Reasoning; Wang et al., Self-Consistency Improves Chain of Thought Reasoning in Language Models; Madaan et al., Self-Refine: Iterative Refinement with Self-Feedback.
- Software: NumPy, SciPy, pandas, PyArrow, Matplotlib, PyTorch, Hugging Face Transformers/Datasets/Hub/Tokenizers/Accelerate, and Safetensors. Versions are saved in the frozen configurations.

The implementation, experiment packaging, analysis and notebook were developed with ChatGPT assistance. The author should review the final submission and provide any AI-use disclosure required by the course policy.
