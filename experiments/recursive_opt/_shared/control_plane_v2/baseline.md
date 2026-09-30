# Control plane v2 baseline

- Captured: 2026-08-21T19:37:14+02:00
- Repository: `/home/xav/code/Trace`
- Conda environment: `humanllm`
- SHA: `6fc278a398709fe79a0fc9be22bae99bffd8cba6`
- Baseline state: **FAILED** because the complete suite could not collect `tests/unit_tests/test_recursive_opt_abc_probe.py`.
- Modification policy: no production or test code was changed before or after these commands; only phase-0 evidence artifacts were added.

## Repository identity

### `git status --short`

Exit: 0

```text
?? .claude/
?? .codex
?? CLAUDE.md
?? XP_recurse_2/
?? examples/XP_1stattempt/
?? examples/XP_recurse_2/
?? examples/examples/
?? examples/notebook_outputs/recursive_opt_abc_probe/
?? examples/notebook_outputs/recursive_opt_use_cases/_probes/
?? examples/notebook_outputs/recursive_opt_use_cases/guarded_decisions_20260624_002620/
?? examples/notebook_outputs/recursive_opt_use_cases/optimizer_compare_uc_opv2_vs_opm4_random.json
?? examples/notebook_outputs/recursive_opt_use_cases/recursive_opt_use_cases_uc_opm4_random.executed.ipynb
?? examples/notebook_outputs/recursive_opt_use_cases/recursive_opt_use_cases_uc_opv2_resume.executed.ipynb
?? examples/notebook_outputs/recursive_opt_use_cases/three_way_live_20260622_082230/
?? examples/notebook_outputs/recursive_opt_use_cases/three_way_live_20260622_fixed_124852/
?? examples/notebook_outputs/recursive_opt_use_cases/three_way_n3_live_20260624_100922/
?? examples/notebook_outputs/recursive_opt_use_cases/three_way_n5_20260624_082016/
?? examples/notebook_outputs/recursive_opt_use_cases/three_way_n5_20260624_082329/
?? examples/notebook_outputs/recursive_opt_use_cases/three_way_n5_rest_20260624_094718/
?? examples/notebook_outputs/recursive_opt_use_cases/three_way_stage2_20260624_150102/
?? examples/notebook_outputs/recursive_opt_use_cases/tier_continuation_summary_latest.json
?? examples/notebook_outputs/recursive_opt_use_cases/tier_run_summary_latest.json
?? examples/notebook_outputs/recursive_opt_use_cases/uc11_rootcause_pf_20260623_170557/
?? examples/notebook_outputs/recursive_opt_use_cases/uc14_only_psmulti_opm4_rolling_20260703_142251/
?? examples/notebook_outputs/recursive_opt_use_cases/uc14_psmulti_opm4_rolling_20260703_141713/
?? examples/notebook_outputs/recursive_opt_use_cases/uc14_selected_run_summary_latest.json
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_inner6_N24_seed0_opm4_rolling_20260704_010108/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_inner6_N24_seed0_opm_multi_llm_rolling_20260704_012405/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_inner6_N24_seed0_opv2_20260703_151937/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_inner6_N24_seed0_opv2_recursive_resume_20260704_005222/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_inner6_N24_seed0_outer_optimizer_compare_latest.json
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_inner6_N24_seed0_outer_optimizer_compare_with_multi_llm_latest.json
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_m3_outer_optimizer_compare_20260704_165615/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_m3_outer_optimizer_compare_timeout_20260704_211713/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_opm_multi_equal_steps_20260701_230643/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_opm_multi_equal_steps_fast_20260701_231325/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_opm_multi_pilot_20260701_230100/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_opm_multi_random_20260702_074717/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_opm_multi_rolling_20260702_082238/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_opm_multi_rolling_clean_20260702_151930/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_opm_multi_same_steps_20260701_235654/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_opm_multi_same_steps_rerun_20260702_005938/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_outer_opv2_common_inner_opv2_N24_n3_20260703_151257/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_outer_psmulti_opm4_rolling_20260703_145904/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_pilot_outer_psmulti_opm4_rolling_20260703_150320/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_psmulti_opm4_rolling_20260703_145042/
?? examples/notebook_outputs/recursive_opt_use_cases/uc4_single_skeptical_expert_20260702_072615/
?? examples/notebook_outputs/recursive_opt_use_cases/uc_opm4_random/
?? examples/notebook_outputs/recursive_opt_use_cases/uc_opv2/
?? examples/notebook_outputs/recursive_opt_use_cases/uc_rootcause_pf_20260623_165657/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_20260614_110901/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_20260614_121931/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_20260614_140804/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_20260614_190736/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_20260625_151531/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_20260626_141706/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_frontier_20260614/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_frontier_uc8_probe/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_frontier_v2_20260614/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_frontier_v3_20260614/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_frontier_v4_20260614/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_frontier_v5_20260615/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_full_live_20260619_000000/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_full_live_20260619_010000/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_full_live_20260619_021500/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_full_live_20260619_socket_blocked_after_value_fixes/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_full_live_20260619_socket_blocked_retry2/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_full_live_20260620_003000/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_full_live_20260620_033000/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_full_live_20260620_next_actions/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_full_live_20260620_next_actions_clean/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_full_live_20260622_231358/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_live_20260613_215505/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_live_deep_20260614_000827/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_live_deep_budgeted_20260614_012157/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_live_deep_fixed_20260614_004207/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_live_fixed_20260613_221437/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_n5_continuation_20260623_131535/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_n5_norm_20260623_112707/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_n5_threeway_20260623_114900/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_n5_uc3_retry_20260623_155010/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_n5_uc3_retry_20260623_155032/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_rootcause_20260614_022600/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_rootcause_final_20260614/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_rootcause_qasper_20260614_032217/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_selected_policy_fix_20260623_082014/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_threeway_policy_fix_20260623_090045/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_tool_policy_fix_20260614/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_uc13_live_20260618_000000/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_uc13_live_fix2_20260618_000000/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_uc13_live_fix_20260618_000000/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_uc13_live_uc13only_20260619_000000/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_uc13_offline_20260618_000000/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_uc14_full_gate_20260630_164228/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_uc14_full_gate_rerun_20260701_191758/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_uc14_stepwise_20260630_150812/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_uc14_target_heavy_nb_20260626_171739/
?? examples/notebook_outputs/recursive_opt_use_cases/use_cases_uc14_underiter_20260630_152322/
?? examples/notebooks/notebook_outputs/
?? memOLD/
?? notebook_outputs/
```

### `git branch --show-current`

Exit: 0

```text
recursive_opt
```

### `git rev-parse HEAD`

Exit: 0

```text
6fc278a398709fe79a0fc9be22bae99bffd8cba6
```

### `git log -5 --oneline`

Exit: 0

```text
6fc278a3 improved explanations in recursive_opt
a8fab200 improved config for Multi expert and Multi LLM optimization
2ad8eb09 restrustured recusrive opt notebook
5a03e208 added OptoPrimeMultiV2
5a148ddb updated UC4 / UC14
```

### Branch ancestry

Command:

```bash
git merge-base --is-ancestor recursive_opt HEAD
```

Exit: 0. The checked-out branch is itself `recursive_opt`; `origin/recursive_opt` is also present.

## Environment

The generic shell did not expose `conda` on `PATH`. Read-only discovery found `/home/xav/miniconda3/bin/conda`, after which all Python commands explicitly sourced `/home/xav/miniconda3/etc/profile.d/conda.sh` and activated `humanllm`.

### `python --version`

Command:

```bash
source /home/xav/miniconda3/etc/profile.d/conda.sh && conda activate humanllm && python --version
```

Exit: 0

```text
Python 3.12.13
```

### `pip freeze`

Command:

```bash
source /home/xav/miniconda3/etc/profile.d/conda.sh && conda activate humanllm && pip freeze
```

Exit: 0

```text
absl-py==2.4.0
aiofiles==24.1.0
aiohappyeyeballs==2.6.1
aiohttp==3.12.15
aiohttp-retry==2.9.1
aiosignal==1.4.0
annotated-doc==0.0.4
annotated-types @ file:///home/task_176174487955274/conda-bld/annotated-types_1761744892998/work
anyio @ file:///home/task_177313441500085/croot/anyio_1773134439506/work
asttokens==3.0.1
attrs @ file:///home/task_177495961277682/croot/attrs_1774959643237/work
beautifulsoup4 @ file:///home/task_177029154460012/croot/beautifulsoup4-split_1770291774176/work
black==26.3.1
bleach @ file:///home/task_176415343735854/conda-bld/bleach_1764153571540/work
blinker==1.9.0
brotlicffi @ file:///home/task_176174500256851/conda-bld/brotlicffi_1761745022533/work
cachetools==6.2.6
certifi @ file:///home/conda/feedstock_root/build_artifacts/certifi_1776866578774/work/certifi
cffi @ file:///home/task_176183215096628/conda-bld/cffi_1761832773152/work
charset-normalizer @ file:///home/task_176174481308776/conda-bld/charset-normalizer_1761744826568/work
claude-agent-sdk==0.1.77
click==8.1.8
comm==0.2.3
contourpy==1.3.3
cryptography==48.0.0
cuda-bindings==13.2.0
cuda-pathfinder==1.5.4
cuda-toolkit==13.0.2
cycler==0.12.1
dacite==1.9.2
dataclasses-json==0.6.7
datasets==4.8.4
daytona==0.173.0
daytona_api_client==0.173.0
daytona_api_client_async==0.173.0
daytona_toolbox_api_client==0.173.0
daytona_toolbox_api_client_async==0.173.0
debugpy==1.8.20
decorator==5.2.1
defusedxml @ file:///tmp/build/80754af9/defusedxml_1615228127516/work
Deprecated==1.3.1
deprecation==2.1.0
dill==0.4.1
dirhash==0.5.0
distro @ file:///croot/distro_1714488253808/work
# Editable install with no version control (document_embedding_analysis==0.1.0)
-e /home/user/code/document_embedding_analysis
einops==0.8.2
evaluate==0.4.6
executing==2.2.1
faiss-cpu==1.13.2
fastapi==0.136.1
fastjsonschema @ file:///home/task_176371856301416/conda-bld/python-fastjsonschema_1763718583388/work
fastuuid==0.14.0
filelock==3.29.0
Flask==3.1.3
fonttools==4.62.1
frozenlist==1.8.0
fsspec==2026.2.0
googleapis-common-protos==1.75.0
graphviz==0.21
greenlet==3.5.0
grpcio==1.80.0
h11 @ file:///home/task_176193127072780/conda-bld/h11_1761931281921/work
h2==4.3.0
harbor==0.6.6
hf-xet==1.4.3
hpack==4.1.0
html5lib @ file:///Users/ktietz/demo/mc3/conda-bld/html5lib_1629144453894/work
httpcore @ file:///croot/httpcore_1748526048470/work
httpx @ file:///home/task_176044736830261/conda-bld/httpx_1760447380560/work
httpx-sse==0.4.3
huggingface_hub==1.14.0
hyperframe==6.1.0
idna @ file:///home/task_176191177554882/conda-bld/idna_1761911981359/work
importlib_metadata==8.5.0
iniconfig==2.3.0
ipykernel==7.2.0
ipython==9.13.0
ipython_pygments_lexers==1.1.1
itsdangerous==2.2.0
jedi==0.20.0
Jinja2 @ file:///croot/jinja2_1741710844255/work
jiter @ file:///home/task_176289732700212/conda-bld/jiter_1762897373249/work
joblib==1.5.3
jsonpatch @ file:///croot/jsonpatch_1714483231291/work
jsonpointer @ file:///home/task_177443289874705/croot/jsonpointer_1774432927391/work
jsonschema==4.23.0
jsonschema-specifications @ file:///home/task_176278841581006/conda-bld/jsonschema-specifications_1762788429133/work
jupyter_client==8.8.0
jupyter_core==5.9.1
jupyterlab_pygments @ file:///croot/jupyterlab_pygments_1741124142640/work
kiwisolver==1.5.0
langchain-classic==1.0.6
langchain-community==0.4.1
langchain-core==1.3.3
langchain-openai @ file:///croot/langchain-openai_1749072012489/work
langchain-protocol==0.0.15
langchain-text-splitters==1.1.2
langgraph @ file:///home/task_177012822436410/croot/langgraph_1770129091385/work
langgraph-checkpoint @ file:///home/task_176129728282744/conda-bld/langgraph-checkpoint_1761297703465/work
langgraph-prebuilt @ file:///home/task_177011644308761/croot/langgraph-prebuilt_1770117306030/work
langgraph-sdk @ file:///home/task_176907542752910/croot/langgraph-sdk_1769075447928/work
langsmith @ file:///home/task_175948403537300/conda-bld/langsmith_1759484056482/work
litellm==1.75.0
loguru==0.7.3
Markdown==3.10.2
markdown-it-py==4.0.0
MarkupSafe @ file:///croot/markupsafe_1738584038848/work
marshmallow==3.26.2
matplotlib==3.10.9
matplotlib-inline==0.2.2
mauve-text==0.4.0
mcp==1.27.0
mdurl==0.1.2
memory-profiler==0.61.0
mistune @ file:///croot/mistune_1741124011532/work
mmh3==5.2.1
mpmath==1.3.0
multidict==6.7.1
multiprocess==0.70.19
mypy_extensions==1.1.0
nbclient @ file:///home/task_177157713402246/croot/nbclient_1771577174483/work
nbconvert @ file:///home/task_177212134708836/croot/nbconvert-meta_1772121377546/work
nbformat @ file:///croot/nbformat_1728049424075/work
nest-asyncio==1.6.0
networkx==3.6.1
nltk==3.9.4
numpy==2.4.4
nvidia-cublas==13.1.0.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.19.0.56
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.0
nvidia-nccl-cu13==2.28.9
nvidia-nvjitlink==13.0.88
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
obstore==0.8.2
openai==2.24.0
openevolve==0.2.27
opentelemetry-api==1.41.1
opentelemetry-exporter-otlp-proto-common==1.41.1
opentelemetry-exporter-otlp-proto-http==1.41.1
opentelemetry-instrumentation==0.62b1
opentelemetry-instrumentation-aiohttp-client==0.62b1
opentelemetry-proto==1.41.1
opentelemetry-sdk==1.41.1
opentelemetry-semantic-conventions==0.62b1
opentelemetry-util-http==0.62b1
orjson @ file:///home/task_176255012329183/conda-bld/orjson_1762550841496/work
ormsgpack @ file:///home/task_176129338032761/conda-bld/ormsgpack_1761294276257/work
packaging==26.2
pandas==3.0.2
pandocfilters @ file:///home/task_175697769885483/conda-bld/pandocfilters_1756977745951/work
parso==0.8.7
pathspec==1.0.4
pdfminer.six==20260107
pexpect==4.9.0
pillow==12.2.0
platformdirs==4.9.6
pluggy==1.6.0
postgrest==2.30.0
prompt_toolkit==3.0.52
propcache==0.4.1
protobuf==6.33.6
psutil==7.2.2
ptyprocess==0.7.0
pure_eval==0.2.3
pyarrow==23.0.1
pycparser @ file:///home/task_177495967481921/croot/pycparser_1774959706730/work
pydantic @ file:///home/task_177313435524461/croot/pydantic_1773134393146/work
pydantic-settings==2.14.0
pydantic_core @ file:///home/task_176400943780574/conda-bld/pydantic-core_1764009744472/work
Pygments==2.20.0
pyiceberg==0.11.1
PyJWT==2.12.1
pypandoc==1.17
pyparsing==3.3.2
pyroaring==1.1.0
pyserial==3.5
PySocks @ file:///home/task_176175225699803/conda-bld/pysocks_1761753009944/work
pytest==9.0.3
python-dateutil==2.9.0.post0
python-dotenv==1.2.2
python-multipart==0.0.27
pytokens==0.4.1
PyYAML @ file:///croot/pyyaml_1728657952215/work
pyzmq==27.1.0
realtime==2.30.0
referencing @ file:///home/task_176253675477096/conda-bld/referencing_1762536887597/work
regex==2026.4.4
requests @ file:///home/task_177506595771297/croot/requests_1775066062226/work
requests-toolbelt @ file:///work/perseverance-python-buildout/croot/requests-toolbelt_1698846872000/work
rich==14.3.4
rouge_score==0.1.2
rpds-py @ file:///home/task_176236637726140/conda-bld/rpds-py_1762366483045/work
ruff==0.15.12
safetensors==0.7.0
scantree==0.0.4
scikit-learn==1.8.0
scipy==1.17.1
sentence-transformers==5.4.1
setuptools==82.0.1
shellingham==1.5.4
shortuuid==1.0.13
six==1.17.0
sniffio @ file:///home/task_176432905353501/conda-bld/sniffio_1764329151381/work
soupsieve @ file:///work/perseverance-python-buildout/croot/soupsieve_1698866207280/work
SQLAlchemy==2.0.49
sse-starlette==3.4.2
stack-data==0.6.3
starlette==1.0.0
storage3==2.30.0
StrEnum==0.4.15
strictyaml==1.7.3
supabase==2.30.0
supabase-auth==2.30.0
supabase-functions==2.30.0
sympy==1.14.0
tenacity @ file:///home/task_176718418080888/croot/tenacity_1767184774484/work
tensorboard==2.20.0
tensorboard-data-server==0.7.2
tensorboardX==2.6.5
threadpoolctl==3.6.0
tiktoken==0.12.0
tinycss2 @ file:///croot/tinycss2_1738337643607/work
tokenizers==0.22.2
toml==0.10.2
torch==2.11.0
tornado==6.5.5
tqdm @ file:///home/task_177139829076527/croot/tqdm_1771398703645/work
-e git+https://github.com/doxav/Trace-Bench.git@716cf82674d0dcb6a62c3f7b72a87b571bf35aa1#egg=trace_bench
-e git+https://github.com/doxav/NewTrace.git@533570f490c2bc3b719f3a3487c3d5331d88f06e#egg=trace_opt
traitlets==5.15.0
transformers==5.8.0
triton==3.6.0
typer==0.23.1
typing-inspect==0.9.0
typing-inspection @ file:///home/task_176061388633419/conda-bld/typing-inspection_1760614166163/work
typing_extensions @ file:///croot/typing_extensions_1756280817316/work
urllib3 @ file:///croot/urllib3_1750775463400/work
uuid_utils==0.14.1
uvicorn==0.46.0
wcwidth==0.7.0
webencodings @ file:///work/perseverance-python-buildout/croot/webencodings_1698866454420/work
websockets==15.0.1
Werkzeug==3.1.8
wheel==0.47.0
wrapt==2.1.2
xxhash @ file:///croot/python-xxhash_1737039434400/work
yarl==1.23.0
zipp==3.23.1
zstandard @ file:///croot/zstandard_1731356346222/work
```

## Mandated targeted baseline

Command:

```bash
source /home/xav/miniconda3/etc/profile.d/conda.sh && conda activate humanllm && PYTHONPATH=. pytest -q \
  tests/unit_tests/test_recursive_spec.py \
  tests/unit_tests/test_recursive_opt.py \
  tests/unit_tests/test_recursive_budget_experiments.py \
  tests/unit_tests/test_recursive_field_activation.py \
  tests/unit_tests/test_recursive_numeric_optimizers.py \
  tests/unit_tests/test_recursive_opt_three_way.py \
  tests/unit_tests/test_recursive_opt_traces.py \
  tests/unit_tests/test_objectives.py \
  tests/unit_tests/test_evaluators_vector.py \
  tests/unit_tests/test_trainers_multiobjective.py
```

Exit: 0  
Command wall time: 10.247042739 seconds  
Pytest duration: 8.33 seconds

```text
........................................................................ [ 30%]
.............s..s....................................................... [ 61%]
........................................................................ [ 91%]
....................                                                     [100%]
234 passed, 2 skipped in 8.33s
```

## Complete available suite

Command:

```bash
source /home/xav/miniconda3/etc/profile.d/conda.sh && conda activate humanllm && PYTHONPATH=. pytest -q
```

Exit: 2  
Command wall time: 9.490184091 seconds  
Pytest duration before collection stopped: 8.15 seconds

```text
==================================== ERRORS ====================================
______ ERROR collecting tests/unit_tests/test_recursive_opt_abc_probe.py _______
ImportError while importing test module '/home/xav/code/Trace/tests/unit_tests/test_recursive_opt_abc_probe.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../miniconda3/envs/humanllm/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit_tests/test_recursive_opt_abc_probe.py:7: in <module>
    from examples import recursive_opt_abc_probe as probe
examples/recursive_opt_abc_probe.py:23: in <module>
    from opto.features.graph import LangGraphAdapter
E   ImportError: cannot import name 'LangGraphAdapter' from 'opto.features.graph' (unknown location)
=============================== warnings summary ===============================
tests/llm_optimizers_tests/test_gepa_benchmark.py:54
  /home/xav/code/Trace/tests/llm_optimizers_tests/test_gepa_benchmark.py:54: PytestUnknownMarkWarning: Unknown pytest.mark.slow - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.slow

../../miniconda3/envs/humanllm/lib/python3.12/site-packages/langgraph/cache/base/__init__.py:8
  /home/xav/miniconda3/envs/humanllm/lib/python3.12/site-packages/langgraph/cache/base/__init__.py:8: LangChainPendingDeprecationWarning: The default value of `allowed_objects` will change in a future version. Pass an explicit value (e.g., allowed_objects='messages' or allowed_objects='core') to suppress this warning.
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
ERROR tests/unit_tests/test_recursive_opt_abc_probe.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
2 warnings, 1 error in 8.15s
```

## Stop decision

The objective explicitly requires stopping when the baseline fails. The failure predates any patch and is consistent with the objective's warning that the graph package/contracts may be missing. No attempt was made to correct or suppress it.

## Continuation recheck

Rechecked on 2026-08-21 at the unchanged SHA `6fc278a398709fe79a0fc9be22bae99bffd8cba6` with no tracked worktree changes:

```bash
source /home/xav/miniconda3/etc/profile.d/conda.sh && conda activate humanllm && PYTHONPATH=. pytest -q tests/unit_tests/test_recursive_opt_abc_probe.py --collect-only
```

Result: exit 2, no tests collected, with the same `ImportError` for `LangGraphAdapter` from `opto.features.graph`. The stop condition therefore remains active.

## User-provided graph import recheck

The user subsequently staged `opto/features/graph/` from the `experimental` branch (five files, 1,019 added lines) and requested another baseline check. At the unchanged commit SHA, with those staged files present:

- Targeted baseline: exit 0; `234 passed, 2 skipped in 6.83s`; command wall time 8.255802963 seconds.
- Complete suite: exit 2 during collection after 7.17 seconds; command wall time 8.428564531 seconds.

The original missing export is now present, but collection reaches the next absent dependency:

```text
opto/features/graph/adapter.py:18: in <module>
    from opto.trace.io.bindings import Binding
E   ModuleNotFoundError: No module named 'opto.trace.io.bindings'
```

`graph_instrumentation.py` additionally imports the absent `opto.trace.io.observers`. Importing these dependencies would modify Trace core, and the wholesale 1,019-line graph copy conflicts with the requirement to extract only the minimal graph contracts. No further files were imported or patched.

## User-provided Trace IO import recheck

The user then staged the matching `opto/trace/io/` folder. Together, the user-provided graph and Trace IO imports add 5,023 staged lines across 17 files.

- Targeted baseline: exit 0; `236 passed in 7.04s`; command wall time 8.483390334 seconds.
- Complete suite: collection succeeded, then recorded 16 failures before entering a live outbound HTTPS retry. It was interrupted after 250.69 seconds to avoid uncontrolled network/paid calls, as live LLM tests are required to remain manual.

Representative complete-suite failures were rerun in a detached temporary worktree at the pristine SHA, without any imported graph/Trace IO files:

```bash
PYTHONPATH=. pytest -q tests/features_tests/test_flows_compose.py::test_tracedllm_and_optoprimev2_prompt_with_mock_llm
PYTHONPATH=. pytest -q tests/llm_optimizers_tests/test_opro_v2.py::test_tag_template_change tests/llm_optimizers_tests/test_opro_v2.py::test_extraction_pipeline
PYTHONPATH=. pytest -q tests/llm_optimizers_tests/test_bbh_subset.py --maxfail=1
```

Results at pristine SHA: respectively 1 failed, 2 failed, and 1 failed with the same error signatures. These failures therefore predate the control-plane patch and are out of scope. The explicitly mandated targeted baseline is green, so implementation may proceed while retaining the complete-suite failures as known baseline failures.
