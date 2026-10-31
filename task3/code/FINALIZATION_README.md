# Task 3 CPU finalization

Extract the three Python files into the existing task3/code folder. Keep the frozen scorer, selector, live-generation and configuration files unchanged. Requires completed Task 3 live scoring and Matplotlib 3.10.8 plus the frozen scorer dependencies.

Run:

    python task3/code/finalize_task3.py --source-task-dir task3 --repo-dir pa3_repo

This performs the full saved-record CPU audit, renders eight figures, writes compact/paired result tables and qualitative examples, and generates a clean CPU report notebook. It copies scoped Task 3 source/configuration/split/result/analysis files into pa3_repo/task3 and adds an idempotent Task 3 link section to the top-level README. Task 1/2 files are preserved. Raw parquet, NPZ caches, MathVista metadata and model weights are excluded.

The output notebook is pa3_repo/task3/Task3.ipynb. Upload it to Colab, use a CPU runtime, and Run all. GPU reproduction remains explicitly False. Save/download the executed notebook before publishing. No Git operations, token prompts, Drive uploads or debugging cells are embedded in the notebook.

The finalizer generates a fresh clean notebook each time. Do not rerun it after saving an executed notebook into the repo folder unless you intend to regenerate that notebook. Repository notebook hash is refreshed as part of the later publishing step. The Drive pa3_repo folder need not be a Git checkout; publishing uses a separate fresh clone.

Local validation: saved metrics and paired confidence intervals reproduced for both datasets; all live calls and costs audited; all 13 notebook code cells executed sequentially through IPython on CPU; eight PNG figures inspected; scoped export hashes and Task 1/2 preservation checked. Colab mount and the disabled optional GPU branch require their corresponding environments and were not executed locally.
