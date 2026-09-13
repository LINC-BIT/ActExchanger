# ActExchanger

## Outline

<a href="#1-overview">1. Overview</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#11-introduction">1.1 Introduction</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#12-hardwaresoftware-requirements-and-dependencies">1.2 Hardware/software Requirements and Dependencies</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#121-hardware-requirements">1.2.1 Hardware Requirements</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#122-software-requirements">1.2.2 Software Requirements</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#123-get-source-code">1.2.3 Get Source Code</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#124-install-dependencies">1.2.4 Install Dependencies</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#125-about-dataset">1.2.5 About Dataset</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#13-treatment-measure-for-unusual-behaviors">1.3 Treatment Measure for Unusual Behaviors</a><br>
<a href="#2-evaluation-reproduction">2. Evaluation Reproduction</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#21-one-click-reproduction">2.1 One-click Reproduction</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#22-step-by-step-reproduction">2.2 Step-by-Step Reproduction</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#221-experiment-1-figure-7-in-section-ivb-comparison-of-accuracy-under-dynamic-environment">2.2.1 Experiment 1: (Figure 7 in Section IV.B) Comparison of Accuracy under Dynamic Environment</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#222-experiment-2-figure-9-and-table-i-in-section-ivc-comparison-of-computational-and-communication-costs">2.2.2 Experiment 2: (Figure 9 and Table I in Section IV.C) Comparison of Computational and Communication Costs</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#223-experiment-3-figure-10-in-section-ivd-design-choice-validation-by-ablation">2.2.3 Experiment 3: (Figure 10 in Section IV.D) Design Choice Validation by Ablation</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#224-experiment-4-discussion-1-in-section-ive-icl-ability">2.2.4 Experiment 4: (Discussion 1 in Section IV.E) ICL Ability</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#225-experiment-5-discussion-2-in-section-ive-comparison-between-vla-models-and-cnns">2.2.5 Experiment 5: (Discussion 2 in Section IV.E) Comparison between VLA Models and CNNs</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#226-experiment-6-discussion-3-in-section-ive-maximum-supported-model-size">2.2.6 Experiment 6: (Discussion 3 in Section IV.E) Maximum supported model size</a><br>
<a href="#3-addition-of-a-new-vla-model-and-three-types-of-marl-methods">3. Addition of a New VLA Model and Three Types of MARL Methods</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#31-new-vla-model-integration-example-vla-adapter">3.1 New VLA Model Integration Example: VLA-Adapter</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#32-cnn-based-marl-example-happo">3.2 CNN-based MARL Example: HAPPO</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#321-agent-model-selection">3.2.1 Agent Model Selection</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#322-planner-model-interface">3.2.2 Planner Model Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#323-communication-interface">3.2.3 Communication Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#324-continual-online-rl-interface">3.2.4 Continual Online RL Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#33-llm-based-marl-example-roco">3.3 LLM-based MARL Example: ROCO</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#331-agent-model-selection">3.3.1 Agent Model Selection</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#332-planner-model-interface">3.3.2 Planner Model Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#333-communication-interface">3.3.3 Communication Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#334-continual-online-rl-interface">3.3.4 Continual Online RL Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#34-vla-based-marl-example-maple">3.4 VLA-based MARL Example: MAPLE</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#341-agent-model-selection">3.4.1 Agent Model Selection</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#342-communication-interface">3.4.2 Communication Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#343-continual-online-rl-interface">3.4.3 Continual Online RL Interface</a><br>

## 1. Overview


### 1.1 Introduction<img src="./heading-divider.svg" alt="" width="100%" height="1">

- **Background**:

  - **Heterogeneous embodied AI agents**: VLA-based robot arms and dexterous hands may have different embodiments, sensors, action spaces, and model backbones while working on a shared task.
  - **Open multi-agent environments**: These agents must cooperate online as tasks, surroundings, available actions, and the set of participating agents change.

- **Method: action-grained knowledge exchange for multi-agent online RL**:

  - **Local action-specific feature generation**: Each agent obtains features from its local observation and embodiment information, with the features associated with the actions that the agent can perform.
  - **Selective common-action exchange and aggregation**: Agents exchange only knowledge for actions shared by the sender and receiver. The feature aggregator maps heterogeneous peer features into a common representation and fuses them with the local feature before policy learning.
  - **Compact semantic and spatial representation**: ActExchanger uses the MDL representation to store semantic and spatial information in compact action-grained vectors, reducing unnecessary communication between devices.

- **Evaluation**:
  - **Basic setting**: The experiments compare 9 representative MARL methods across 4 typical embodied AI application scenarios.
  - **Major results**: ActExchanger achieves up to 28.06% higher accuracy, 3.73x shorter training time, 9.72x lower communication time, 8.50x less transmission volume, and 4.16x lower energy consumption.

### 1.2 Hardware/software Requirements and Dependencies<img src="./heading-divider.svg" alt="" width="100%" height="1">

#### 1.2.1 Hardware Requirements<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Option 1: Hardware requirements for full running of experiments in the paper**:

  <table align="center">
    <thead>
      <tr>
        <th>RAM</th>
        <th>CPU</th>
        <th>Disk</th>
        <th>GPU</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>128 GB</td>
        <td>One 64-core server CPU (e.g., Intel(R) Xeon(R) Gold 6430)</td>
        <td>At least<br>150 GB free</td>
        <td>One NVIDIA GPU with more than 60 GB VRAM (e.g., A100)</td>
      </tr>
    </tbody>
  </table>

- **Option 2: Six recommended hardware requirements for running minimal working examples**:

  <table align="center">
    <thead>
      <tr>
        <th></th>
        <th>RAM</th>
        <th>CPU</th>
        <th>Disk</th>
        <th>GPU</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Single-CPU server</td>
        <td>16–32 GB</td>
        <td>One 8-core server CPU (e.g., Intel Xeon E-2388G)</td>
        <td>At least 80 GB free</td>
        <td>—</td>
      </tr>
      <tr>
        <td>GPU-equipped server</td>
        <td>32–64 GB</td>
        <td>One 12-core server CPU (e.g., Intel Xeon Silver 4310)</td>
        <td>At least 80 GB free</td>
        <td>One NVIDIA GPU with 8–12 GB VRAM (e.g., NVIDIA RTX 3060)</td>
      </tr>
      <tr>
        <td>CPU-only desktop</td>
        <td>16–32 GB</td>
        <td>One 16-core desktop CPU (e.g., Intel Core i7-13700)</td>
        <td>At least 80 GB free</td>
        <td>Integrated graphics</td>
      </tr>
      <tr>
        <td>GPU-equipped desktop</td>
        <td>16–32 GB</td>
        <td>One 20-core desktop CPU (e.g., Intel Core i7-14700)</td>
        <td>At least 80 GB free</td>
        <td>One NVIDIA GPU with 8 GB VRAM (e.g., NVIDIA RTX 4060)</td>
      </tr>
      <tr>
        <td>CPU-only laptop</td>
        <td>16–32 GB</td>
        <td>One 12-core CPU (e.g., Intel Core Ultra 7 155U)</td>
        <td>At least 80 GB free</td>
        <td>Integrated graphics</td>
      </tr>
      <tr>
        <td>GPU-equipped laptop</td>
        <td>16–32 GB</td>
        <td>One 16-core CPU (e.g., Intel Core i7-14650HX)</td>
        <td>At least 80 GB free</td>
        <td>One NVIDIA GPU with 8 GB VRAM (e.g., RTX 4060)</td>
      </tr>
    </tbody>
  </table>

#### 1.2.2 Software Requirements<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Option 1: Recommended software for full running of experiments in the paper**:

  <table align="center">
    <thead>
      <tr>
        <th>Operating System</th>
        <th>CUDA</th>
        <th>Others</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Ubuntu LTS 22.04.4 LTS</td>
        <td>CUDA 13.0</td>
        <td>Kernel 6.8.0-124-generic<br>Docker 29.2.1</td>
      </tr>
    </tbody>
  </table>
      
- **Option 2: Software requirements for running minimal working examples**:

  <table align="center">
    <thead>
      <tr>
        <th>Operating System</th>
        <th>CUDA</th>
        <th>Others</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Ubuntu LTS 20.04+</td>
        <td>CUDA 12.x+<br>(when using a GPU)</td>
        <td>Kernel 5.4+<br>Docker 29.0+</td>
      </tr>
      <tr>
        <td>Windows 10+</td>
        <td>CUDA 12.x+<br>(when using a GPU)</td>
        <td>Docker 29.0+</td>
      </tr>
      <tr>
        <td>macOS 14+</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>Debian 11+</td>
        <td>CUDA 12.x+<br>(when using a GPU)</td>
        <td>Kernel 5.4+<br>Docker 29.0+</td>
      </tr>
      <tr>
        <td>RHEL 8+</td>
        <td>CUDA 12.x+<br>(when using a GPU)</td>
        <td>Kernel 5.4+<br>Docker 29.0+</td>
      </tr>
    </tbody>
  </table>


#### 1.2.3 Get Source Code<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

  ```bash
  git clone https://github.com/LINC-BIT/ActExchanger.git
  ```

#### 1.2.4 Install Dependencies<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Step 1: Install Docker** ([Example running screenshots](install-step-1-example.md))

  ```bash
  # Add Docker's official GPG key:
  sudo apt update
  sudo apt install ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc

  # Add the repository to Apt sources:
  sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
  Types: deb
  URIs: https://download.docker.com/linux/ubuntu
  Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
  Components: stable
  Architectures: $(dpkg --print-architecture)
  Signed-By: /etc/apt/keyrings/docker.asc
  EOF

  sudo apt update

  sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  sudo systemctl status docker --no-pager
  sudo docker run hello-world
  ```
  

- **Step 2: Install Docker plugin for using CUDA** ([Example running screenshots](install-step-2-example.md))
  
  ```bash
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
  

- **Step 3: Install the required dependencies of this artifact:** ([Example running screenshots](install-step-3-example.md))

  ```bash
  cd <ActExchanger directory>

  # Option 1: 
  # pull the full Docker image (33GB)
  # without installing other dependencies
  bash dep.sh

  # Option 2: 
  # pull the minimum Docker image (100MB)
  # and install other dependencies in the Docker container
  TYPE=100M bash dep.sh
  ```
  

- **Step 4: Check the installation:** ([Example running screenshots](imgs/4.1.png))

  ```bash
  bash start_docker.sh
  python -c "import torch; print(torch.__version__)"
  python -c "import torch; print(torch.cuda.is_available())"
  python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA not available')"
  ```
  


**Figure plotting dependencies:**

```bash
pip install matplotlib==3.10.8 pypdf==6.16.2
```

#### 1.2.5 About Dataset<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">


  We do not use any dataset for training. The data for training is sampled from the ManiSkill Benchmark's environments.

### 1.3 Treatment Measure for Unusual Behaviors

| Unusual behavior | Treatment |
| --- | --- |
| `ModuleNotFoundError` for `mani_skill`, `gymnasium`, `accelerate`, or a VLA model | Activate the training environment containing the selected model and install the missing package. Plotting scripts only need the plotting dependencies listed above. |
| Checkpoint `FileNotFoundError` | Supply the corresponding `RUN_DIR`/`BASELINE_RUN_DIR` or the explicit run-directory arguments. The repository does not include large checkpoints. |
| CUDA out-of-memory | Reduce the environment count, batch size, or rollout length; use the minimal benchmark scripts first; then move to a GPU with sufficient VRAM. |
| No GPU found in a training launcher | Run the CPU-compatible plotting/benchmark command, or configure `CUDA_VISIBLE_DEVICES` before launching a GPU experiment. |
| A plotting script cannot find TensorBoard metrics | Check that the run directory contains `metrics_history.json` or TensorBoard event files and pass the correct run directory. |
| A shell script is run from another directory | Use the `eval/.../run_*.sh` path from the repository root. The wrappers resolve their own repository path before launching Python. |

## 2. Evaluation Reproduction

### 2.1 One-click Reproduction<img src="./heading-divider.svg" alt="" width="100%" height="1">

We provide a one-click script `eval/run.sh` that runs all experiments sequentially and produces all resulting figures and tables.

- **Option 1: Minimum working examples (completed within 1 day, 20GB GPU memory, and 30GB disk space)**
  ```bash
  cd <ActExchanger directory>
  bash start_docker.sh
  cd <ActExchanger directory in the container>/eval
  MWE=1 bash run.sh
  ```
- **Option 2: Full run (completed within 15 days, 75GB GPU memory, 150GB disk space)**
  ```bash
  cd <ActExchanger directory>
  bash start_docker.sh
  cd <ActExchanger directory in the container>/eval
  bash run.sh
  ```

### 2.2 Step-by-Step Reproduction<img src="./heading-divider.svg" alt="" width="100%" height="1">

Start the container:

```bash
cd <ActExchanger directory>
bash start_docker.sh
cd <ActExchanger directory in the container>/eval
```

#### Notes in the Reproduction<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Clean previous results before running an experiment:**

  ```bash
  bash ../clean_results.sh
  bash ../clean_results.sh --results-root /home/Maniskill/ckpt --force
  ```

  The first command previews the files; `--force` deletes them. Pretraining and SFT checkpoints are preserved.

#### 2.2.1 Experiment 1: (Figure 7 in Section IV.B) Comparison of Accuracy under Dynamic Environment<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Option 1: Commands for full run (200 hours, 75GB GPU memory, 150GB disk space):**
  ```bash
  cd acc_comparison
  bash run_acc_comparison.sh
  ```
- **Option 2: Commands for minimum working examples (2.5 hours, 20GB GPU memory, 60GB disk space):**
  ```bash
  cd acc_comparison
  MWE=1 bash run_acc_comparison.sh
  ```

#### 2.2.2 Experiment 2: (Figure 9 and Table I in Section IV.C) Comparison of Computational and Communication Costs<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Option 1: Commands for full run (200 hours, 75GB GPU memory, 150GB disk space):**
  ```bash
  cd overhead
  bash run_overhead.sh
  ```
- **Option 2: Commands for minimum working examples (2.5 hours, 20GB GPU memory, 60GB disk space):**
  ```bash
  cd overhead
  MWE=1 bash run_overhead.sh
  ```

#### 2.2.3 Experiment 3: (Figure 10 in Section IV.D) Design Choice Validation by Ablation<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Option 1: Commands for full run (80 hours, 128GB GPU memory, 150GB disk space):**
  ```bash
  cd ablation
  bash run_ablation.sh
  ```
- **Option 2: Commands for minimum working examples (40 minutes, 20GB GPU memory, 30GB disk space):**
  ```bash
  cd ablation
  MWE=1 bash run_ablation.sh
  ```

#### 2.2.4 Experiment 4: (Discussion 1 in Section IV.E) ICL Ability<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Option 1: Commands for full run (30 minutes, 20GB GPU memory, 30GB disk space):**
  ```bash
  cd discussion
  RICL_RUN_DIR_OBJECT_PICKING_PLACING=<ricl_run> \
  RICL_RUN_DIR_OBJECT_STACKING=<ricl_run> \
  RICL_RUN_DIR_CUCUMBER_PLACING=<ricl_run> \
  RICL_RUN_DIR_CYLINDER_TRANSFER=<ricl_run> \
  bash run_discussion.sh icl
  ```
- **Option 2: Commands for minimum working examples (10 minutes, 20GB GPU memory, 30GB disk space):**
  ```bash
  cd discussion
  MWE=1 bash run_discussion.sh icl
  ```

#### 2.2.5 Experiment 5: (Discussion 2 in Section IV.E) Comparison between VLA Models and CNNs<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Option 1: Commands for full run (60 minutes, 75GB GPU memory, 30GB disk space):**
  ```bash
  cd discussion
  VLA_RUN_DIR=<vla_run> CNN_RUN_DIR=<cnn_run> bash run_discussion.sh vla-vs-cnn
  ```
- **Option 2: Commands for minimum working examples (20 minutes, 20GB GPU memory, 30GB disk space):**
  ```bash
  cd discussion
  MWE=1 bash run_discussion.sh vla-vs-cnn
  ```

#### 2.2.6 Experiment 6: (Discussion 3 in Section IV.E) Maximum supported model size<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Option 1: Commands for full run (90 minutes, 80GB GPU memory, 128GB disk space):**
  ```bash
  cd discussion
  MODEL_SIZE_RUN_DIRS=<run_1>,<run_2>,... bash run_discussion.sh model-size
  ```
- **Option 2: Commands for minimum working examples (10 minutes, 20GB GPU memory, 30GB disk space):**
  ```bash
  cd discussion
  MWE=1 bash run_discussion.sh model-size
  ```


## 3. Addition of a New VLA Model and Three Types of MARL Methods

### 3.1 New VLA Model Integration Example: VLA-Adapter<img src="./heading-divider.svg" alt="" width="100%" height="1">


  - **Step 1:** Implement `VLAModelInterface`, the VLA-specific extension of `AgentModelInterface`.

    ```python
    import importlib
    from pathlib import Path
    from types import SimpleNamespace

    from api.vla_model_interface import VLAModelInterface

    class VLAAdapter(VLAModelInterface):
        def __init__(self, model_dir, agent_spec, *, backend_module):
            self.model_dir = Path(model_dir)
            self._agent_spec = agent_spec
            self.backend_module = backend_module

        @property
        def model_name(self):
            return "vla_adapter_smolvla"

        @property
        def agent_spec(self):
            return self._agent_spec

        @property
        def policy_class(self):
            backend = importlib.import_module(self.backend_module)
            return backend.MultiAgentVLAAdapterAgent
    ```

  - **Step 2:** Describe the participating agents and their state/action spaces with `VLAAgentSpec`. These dimensions must match the workload's `get_agent_info()` output.

    ```python
    spec = VLAAgentSpec(
        agent_names=("agent-0", "agent-1"),
        state_dims={"agent-0": state_dim, "agent-1": state_dim},
        action_dims={"agent-0": action_dim, "agent-1": action_dim},
        global_state_dim=global_state_dim,
    )
    ```

  - **Step 3:** Bind model construction, observation conversion, action/value inference, trainable-module selection, and optimizer creation to the VLA-Adapter backend.

    ```python
    def build_policy(self, *, device, config):
        names = list(self.agent_spec.agent_names)
        return self.policy_class(
            agent_names=names,
            state_dim=self.agent_spec.state_dims[names[0]],
            global_state_dim=self.agent_spec.global_state_dim,
            action_dim=self.agent_spec.action_dims[names[0]],
            model_dir=self.model_dir,
            freeze_vla_backbone=config.get("freeze_vla_backbone", False),
        ).to(device)

    def build_batch_from_obs(self, obs, *, device):
        batch = self._backend().build_batch_from_obs(
            obs, list(self.agent_spec.agent_names)
        )
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def get_action_and_value(self, policy, batch, *, actions_input=None, deterministic=False):
        return policy.get_action_and_value(dict(batch), actions_input=actions_input)

    def get_action(self, policy, batch, *, deterministic=False):
        return policy.get_action(dict(batch), deterministic=deterministic)

    def get_value(self, policy, batch):
        return policy.get_value(dict(batch))

    def configure_trainable_modules(self, policy, *, freeze_vla_backbone):
        policy.configure_trainable_modules(freeze_vla_backbone)

    def build_optimizer(self, policy, *, config):
        return self._backend().build_optimizer(SimpleNamespace(**config), policy)
    ```

  Complete implementation: [`vla_adapter_smolvla_impl.py`](./api/vla_model_interface_examples/vla_adapter_smolvla_impl.py).

  - **Step 4:** Instantiate the adapter with the selected workload backend.

    ```python
    vla_adapter = VLAAdapter(
        model_dir="ckpt/vla_adapter_smolvla",
        agent_spec=spec,
        backend_module=backend_module,
    )
    ```

### 3.2 CNN-based MARL Example: HAPPO<img src="./heading-divider.svg" alt="" width="100%" height="1">

#### 3.2.1 Agent Model Selection<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

  - **Step 1:** Select an `AgentModelInterface` implementation. This example reuses the `VLAAdapter` configured in Section 3.1.

    ```python
    from api.marl_method_interface import AgentModelInterface

    # The VLAAdapter instance created in Section 3.1, Step 4.
    agent_model: AgentModelInterface = vla_adapter
    ```

#### 3.2.2 Planner Model Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

  - **Step 1:** Create a planner adapter inherited from `PlannerModelInterface`.

    ```python
    import torch

    from api.marl_method_interface import PlannerModelInterface
    from api.marl_method_interface_examples._common import ZeroPlanner

    class HAPPOPlannerModel(PlannerModelInterface):
        planner_name = "happo_planner"

        def build_planner(self, *, agent_model, device, config):
            return ZeroPlanner(agent_model.agent_names).to(device)

        def plan(self, planner, batch, *, deterministic=False):
            return planner(batch)

        def get_action_and_value(
            self, planner, batch, *, actions_input=None, deterministic=False
        ):
            output = planner(batch)
            values = torch.zeros(next(iter(output.values())).shape[0])
            return output, {}, {}, values

        def build_optimizer(self, planner, *, config):
            return torch.optim.Adam(
                planner.parameters(), lr=config.get("learning_rate", 3e-4)
            )

        def update(self, planner, optimizer, rollout, *, config):
            output = planner(rollout["batch"])
            losses = [
                torch.nn.functional.mse_loss(output[name], rollout["planner_targets"][name])
                for name in output
            ]
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return {"planner_loss": float(loss.detach())}
    ```

  - **Step 2:** Initialize the planner with the selected agent model.

    ```python
    planner_impl = HAPPOPlannerModel()
    planner = planner_impl.build_planner(
        agent_model=agent_model,
        device=device,
        config=config,
    )
    ```

  - **Step 3:** Use the planner output as the condition for the agent model before selecting low-level actions.

    ```python
    planner_output = planner_impl.plan(
        planner,
        batch,
        deterministic=False,
    )
    ```

#### 3.2.3 Communication Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

  - **Step 1:** Create a communication adapter inherited from `CommunicationInterface`.

    ```python
    from api.marl_method_interface import CommunicationInterface
    from api.marl_method_interface_examples._common import TensorCommunication

    class HAPPOCommunication(CommunicationInterface):
        communication_name = "happo_tensor_mean"

        def __init__(self):
            self._impl = TensorCommunication()

        def encode_message(self, sender, receiver, feature, *, action_mask=None):
            return self._impl.encode_message(
                sender, receiver, feature, action_mask=action_mask
            )

        def decode_message(self, message, *, receiver, device):
            return self._impl.decode_message(message, receiver=receiver, device=device)

        def aggregate(self, local_feature, remote_features, *, batch=None):
            return self._impl.aggregate(local_feature, remote_features, batch=batch)

        def transmission_size(self, message):
            return self._impl.transmission_size(message)
    ```

  Complete implementation: [`happo_impl.py`](./api/marl_method_interface_examples/happo_impl.py).

  - **Step 2:** Use the communication adapter during agent interaction.

    ```python
    communication = HAPPOCommunication()
    message = communication.encode_message(
        sender="agent_0",
        receiver="agent_1",
        feature=local_feature,
        action_mask=action_mask,
    )
    received_feature = communication.decode_message(
        message,
        receiver="agent_1",
        device=device,
    )
    fused_feature = communication.aggregate(
        local_feature,
        [received_feature],
    )
    ```

#### 3.2.4 Continual Online RL Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

  - **Step 1:** Implement HAPPO rollout and update callbacks with the workload driver signatures.

    ```python
    def happo_collect_rollout(
        *, args, agent, collate_fn, envs, next_obs, next_done,
        accelerator, writer, global_step,
    ):
        ...

    def happo_update_on_policy(
        args, agent, optimizer, data, collate_fn, accelerator,
        stage, writer, rollouts_num_aft_env_change=None,
    ):
        ...
    ```

  - **Step 2:** Bind HAPPO to the selected agent model and call the shared Online RL entry.

    ```python
    from api.marl_online_rl_interface import CallbackMARLOnlineRL, run_continual_online_rl
    from workload.online_driver import run_online_training

    marl_method = CallbackMARLOnlineRL(
        algorithm_name="happo",
        model_name=agent_model.model_name,
        collect_rollout_fn=happo_collect_rollout,
        update_fn=happo_update_on_policy,
    )
    run_continual_online_rl(
        args,
        agent_model=agent_model,
        marl_method=marl_method,
        driver=run_online_training,
    )
    ```

### 3.3 LLM-based MARL Example: ROCO<img src="./heading-divider.svg" alt="" width="100%" height="1">

#### 3.3.1 Agent Model Selection<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

  - **Step 1:** Select an `AgentModelInterface` implementation. This example reuses the `VLAAdapter` configured in Section 3.1.

    ```python
    from api.marl_method_interface import AgentModelInterface

    # The VLAAdapter instance created in Section 3.1, Step 4.
    agent_model: AgentModelInterface = vla_adapter
    ```

#### 3.3.2 Planner Model Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

  - **Step 1:** Create a planner adapter inherited from `PlannerModelInterface`.

    ```python
    import torch

    from api.marl_method_interface import PlannerModelInterface
    from api.marl_method_interface_examples.roco_impl import _RoCoPlanner

    class ROCOPlannerModel(PlannerModelInterface):
        planner_name = "roco_task_planner"

        def build_planner(self, *, agent_model, device, config):
            backend = config.get("llm_planner_backend")
            return _RoCoPlanner(agent_model.agent_names, backend=backend).to(device)

        def plan(self, planner, batch, *, deterministic=False):
            return planner(batch)

        def get_action_and_value(
            self, planner, batch, *, actions_input=None, deterministic=False
        ):
            output = planner(batch)
            first_agent = planner.agent_names[0]
            size = batch[f"agent_states_{first_agent}"].shape[0]
            value = torch.zeros(size, device=next(planner.parameters()).device)
            return output, {}, {}, value

        def build_optimizer(self, planner, *, config):
            return torch.optim.Adam(
                planner.parameters(),
                lr=config.get("planner_learning_rate", 3e-4),
            )

        def update(self, planner, optimizer, rollout, *, config):
            feedback = rollout.get("environment_feedback", "")
            valid = bool(rollout.get("plan_valid", not bool(feedback)))
            return {
                "plan_valid": float(valid),
                "requires_replanning": float(not valid),
            }
    ```

  - **Step 2:** Initialize the planner with the selected agent model and LLM backend.

    ```python
    planner_impl = ROCOPlannerModel()
    planner = planner_impl.build_planner(
        agent_model=agent_model,
        device=device,
        config=config,
    )
    ```

  - **Step 3:** Use the planner decisions to condition each LLM agent before it selects low-level actions.

    ```python
    planner_output = planner_impl.plan(
        planner,
        batch,
        deterministic=False,
    )
    # RoCo-style outputs: dialogue, per-agent sub-tasks, waypoints, and
    # collision/environment feedback used for the next replanning turn.
    dialogue = planner_output["dialogue"]
    subtasks = planner_output["subtasks"]
    waypoints = planner_output["waypoints"]
    ```

#### 3.3.3 Communication Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

  - **Step 1:** Create a communication adapter inherited from `CommunicationInterface`.

    ```python
    from api.marl_method_interface import CommunicationInterface
    from api.marl_method_interface_examples._common import TensorCommunication

    class ROCOCommunication(CommunicationInterface):
        communication_name = "roco_dialogue_message"

        def __init__(self):
            self._impl = TensorCommunication()

        def encode_message(
            self, sender, receiver, feature, *, action_mask=None,
            dialogue_round=0, subtask=None, waypoints=None,
            environment_feedback=None,
        ):
            message = self._impl.encode_message(
                sender, receiver, feature, action_mask=action_mask
            )
            message.update({
                "dialogue_round": dialogue_round,
                "subtask": subtask,
                "waypoints": waypoints,
                "environment_feedback": environment_feedback,
            })
            return message

        def decode_message(self, message, *, receiver, device):
            return self._impl.decode_message(message, receiver=receiver, device=device)

        def aggregate(self, local_feature, remote_features, *, batch=None):
            return self._impl.aggregate(local_feature, remote_features, batch=batch)

        def transmission_size(self, message):
            tensor_bytes = self._impl.transmission_size(message)
            text_bytes = sum(
                len(str(message.get(key, "")).encode("utf-8"))
                for key in ("subtask", "environment_feedback", "dialogue_round")
            )
            return tensor_bytes + text_bytes
    ```

  Complete implementation: [`roco_impl.py`](./api/marl_method_interface_examples/roco_impl.py).

  - **Step 2:** Use the communication adapter to exchange ROCO planning information.

    ```python
    communication = ROCOCommunication()
    message = communication.encode_message(
        sender="agent_0",
        receiver="agent_1",
        feature=local_feature,
        dialogue_round=2,
        subtask="move the left object to the shared staging area",
        waypoints=waypoints["agent_1"],
        environment_feedback="collision risk near the shelf",
    )
    # The message carries RoCo's dialogue round, sub-task, waypoint plan, and
    # environment feedback so the next planner turn can re-plan if needed.
    received_feature = communication.decode_message(
        message,
        receiver="agent_1",
        device=device,
    )
    fused_feature = communication.aggregate(
        local_feature,
        [received_feature],
    )
    ```

#### 3.3.4 Continual Online RL Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

  - **Step 1:** Implement ROCO rollout and update callbacks with the workload driver signatures.

    ```python
    def roco_collect_rollout(
        *, args, agent, collate_fn, envs, next_obs, next_done,
        accelerator, writer, global_step,
    ):
        ...

    def roco_update_on_policy(
        args, agent, optimizer, data, collate_fn, accelerator,
        stage, writer, rollouts_num_aft_env_change=None,
    ):
        ...
    ```

  - **Step 2:** Bind ROCO to the selected agent model and call the shared Online RL entry.

    ```python
    from api.marl_online_rl_interface import CallbackMARLOnlineRL, run_continual_online_rl
    from workload.online_driver import run_online_training

    marl_method = CallbackMARLOnlineRL(
        algorithm_name="roco",
        model_name=agent_model.model_name,
        collect_rollout_fn=roco_collect_rollout,
        update_fn=roco_update_on_policy,
    )
    run_continual_online_rl(
        args,
        agent_model=agent_model,
        marl_method=marl_method,
        driver=run_online_training,
    )
    ```

### 3.4 VLA-based MARL Example: MAPLE<img src="./heading-divider.svg" alt="" width="100%" height="1">

#### 3.4.1 Agent Model Selection<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

  - **Step 1:** Select an `AgentModelInterface` implementation. This example reuses the `VLAAdapter` configured in Section 3.1.

    ```python
    from api.marl_method_interface import AgentModelInterface

    # The VLAAdapter instance created in Section 3.1, Step 4.
    agent_model: AgentModelInterface = vla_adapter
    ```

#### 3.4.2 Communication Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

  - **Step 1:** Create a communication adapter inherited from `CommunicationInterface`.

    ```python
    from api.marl_method_interface import CommunicationInterface
    from api.marl_method_interface_examples._common import TensorCommunication

    class MAPLECommunication(CommunicationInterface):
        communication_name = "maple_latent_tensor"

        def __init__(self):
            self._impl = TensorCommunication()

        def encode_message(self, sender, receiver, feature, *, action_mask=None):
            return self._impl.encode_message(
                sender, receiver, feature, action_mask=action_mask
            )

        def decode_message(self, message, *, receiver, device):
            return self._impl.decode_message(message, receiver=receiver, device=device)

        def aggregate(self, local_feature, remote_features, *, batch=None):
            return self._impl.aggregate(local_feature, remote_features, batch=batch)

        def transmission_size(self, message):
            return self._impl.transmission_size(message)
    ```

  Complete implementation: [`maple_impl.py`](./api/marl_method_interface_examples/maple_impl.py).

  - **Step 2:** Use the communication adapter to exchange MAPLE latent features.

    ```python
    communication = MAPLECommunication()
    message = communication.encode_message(
        sender="agent_0",
        receiver="agent_1",
        feature=local_feature,
    )
    received_feature = communication.decode_message(
        message,
        receiver="agent_1",
        device=device,
    )
    fused_feature = communication.aggregate(
        local_feature,
        [received_feature],
    )
    ```

#### 3.4.3 Continual Online RL Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

  - **Step 1:** Implement MAPLE rollout and update callbacks with the workload driver signatures.

    ```python
    def maple_collect_rollout(
        *, args, agent, collate_fn, envs, next_obs, next_done,
        accelerator, writer, global_step,
    ):
        ...

    def maple_update_on_policy(
        args, agent, optimizer, data, collate_fn, accelerator,
        stage, writer, rollouts_num_aft_env_change=None,
    ):
        ...
    ```

  - **Step 2:** Bind MAPLE to the selected agent model and call the shared Online RL entry.

    ```python
    from api.marl_online_rl_interface import CallbackMARLOnlineRL, run_continual_online_rl
    from workload.online_driver import run_online_training

    marl_method = CallbackMARLOnlineRL(
        algorithm_name="maple",
        model_name=agent_model.model_name,
        collect_rollout_fn=maple_collect_rollout,
        update_fn=maple_update_on_policy,
        rollout_mode="maple",
        train_mode_during_update=True,
    )
    run_continual_online_rl(
        args,
        agent_model=agent_model,
        marl_method=marl_method,
        driver=run_online_training,
    )
    ```
