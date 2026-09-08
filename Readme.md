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
<a href="#3-addition-of-three-types-of-marl-methods">3. Addition of Three Types of MARL Methods</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#31-cnn-based-marl-example-happo">3.1 CNN-based MARL Example: HAPPO</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#311-agent-model-interface">3.1.1 Agent Model Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#312-planner-model-interface">3.1.2 Planner Model Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#313-communication-interface">3.1.3 Communication Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#314-training-loop-interface">3.1.4 Training Loop Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#32-llm-based-marl-example-roco">3.2 LLM-based MARL Example: ROCO</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#321-agent-model-interface">3.2.1 Agent Model Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#322-planner-model-interface">3.2.2 Planner Model Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#323-communication-interface">3.2.3 Communication Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#324-training-loop-interface">3.2.4 Training Loop Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#33-vla-based-marl-example-maple">3.3 VLA-based MARL Example: MAPLE</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#331-agent-model-interface">3.3.1 Agent Model Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#332-communication-interface">3.3.2 Communication Interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#333-training-loop-interface">3.3.3 Training Loop Interface</a><br>

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

  You can obtain the source code for artifacts evaluation by the following command. **The code does not perform any malicious or destructive operations**.

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

- Run the command below to install dependencies for plotting scripts:
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

- **(Recommended) Option 1: Minimum working examples (completed within 1 day and 20GB memory)**
  ```bash
  cd <ActExchanger directory>
  bash start_docker.sh
  cd <ActExchanger directory in the container>/eval
  MWE=1 bash run.sh
  ```
- **Option 2: Full run (completed within 15 days and 60GB memory)**
  ```bash
  cd <ActExchanger directory>
  bash start_docker.sh
  cd <ActExchanger directory in the container>/eval
  bash run.sh
  ```

The reproducing steps of each experiment are described in Section 2.2.

### 2.2 Step-by-Step Reproduction<img src="./heading-divider.svg" alt="" width="100%" height="1">

First of all, run the following command:

```bash
cd <ActExchanger directory>
bash start_docker.sh
cd <ActExchanger directory in the container>/eval
```

Then, run the following commands to reproduce each figure/table in our evaluation.

#### 2.2.1 Experiment 1: (Figure 7 in Section IV.B) Comparison of Accuracy under Dynamic Environment<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Option 1:** Commands for full run on the four multi-agent workloads:
  ```bash
  cd acc_comparison
  bash run_acc_comparison.sh
  ```
- **Option 2:** Commands for minimum working examples:
  ```bash
  cd acc_comparison
  MWE=1 bash run_acc_comparison.sh
  ```

#### 2.2.2 Experiment 2: (Figure 9 and Table I in Section IV.C) Comparison of Computational and Communication Costs<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Option 1:** Commands for full run:
  ```bash
  cd overhead
  bash run_overhead.sh
  ```
- **Option 2:** Commands for minimum working examples:
  ```bash
  cd overhead
  MWE=1 bash run_overhead.sh
  ```

#### 2.2.3 Experiment 3: (Figure 10 in Section IV.D) Design Choice Validation by Ablation<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Option 1:** Commands for full run:
  ```bash
  cd ablation
  bash run_ablation.sh
  ```
- **Option 2:** Commands for minimum working examples:
  ```bash
  cd ablation
  MWE=1 bash run_ablation.sh
  ```

#### 2.2.4 Experiment 4: (Discussion 1 in Section IV.E) ICL Ability<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Option 1:** Commands for full run:
  ```bash
  cd discussion
  bash run_discussion.sh icl
  ```
- **Option 2:** Commands for minimum working examples:
  ```bash
  cd discussion
  MWE=1 bash run_discussion.sh icl
  ```

#### 2.2.5 Experiment 5: (Discussion 2 in Section IV.E) Comparison between VLA Models and CNNs<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Option 1:** Commands for full run:
  ```bash
  cd discussion
  bash run_discussion.sh vla-vs-cnn
  ```
- **Option 2:** Commands for minimum working examples:
  ```bash
  cd discussion
  MWE=1 bash run_discussion.sh vla-vs-cnn
  ```

#### 2.2.6 Experiment 6: (Discussion 3 in Section IV.E) Maximum supported model size<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Option 1:** Commands for full run:
  ```bash
  cd discussion
  bash run_discussion.sh model-size
  ```
- **Option 2:** Commands for minimum working examples:
  ```bash
  cd discussion
  MWE=1 bash run_discussion.sh model-size
  ```


## 3. Addition of Three Types of MARL Methods

### 3.1 CNN-based MARL Example: HAPPO<img src="./heading-divider.svg" alt="" width="100%" height="1">
You can add a new CNN-based MARL baseline according to the following steps.

#### 3.1.1 Agent Model Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Detailed integration steps**:

  - **Step 1:** Create a new file and define `HAPPOAgentModel`, inherited from `AgentModelInterface`.

    ```python
    from api.marl_method_interface import AgentModelInterface

    class HAPPOAgentModel(AgentModelInterface):
        ...
    ```

  - **Step 2:** Define the agent names and the state/action dimensions for the selected workload.

    ```python
    model_impl = HAPPOAgentModel(
        agent_names=("agent_0", "agent_1"),
        state_dims={"agent_0": 32, "agent_1": 32},
        action_dims={"agent_0": 7, "agent_1": 7},
    )
    ```

  - **Step 3:** Implement the CNN policy construction, observation conversion, action sampling, value estimation, and optimizer construction.

    ```python
    def build_model(self, *, device, config):
        # Return a CNN actor-critic model for the selected workload.
        ...

    def build_batch_from_obs(self, obs, *, device):
        # Convert raw ManiSkill observations to the policy batch schema.
        ...

    def get_action_and_value(self, model, batch, *, actions_input=None, deterministic=False):
        # Return per-agent actions, log probabilities, entropies, and values.
        ...
    ```

  - **Step 4:** Initialize the agent model interface.

    ```python
    agent_model_impl = HAPPOAgentModel(
        agent_names=agent_names,
        state_dims=state_dims,
        action_dims=action_dims,
    )
    ```

  - **Step 5:** Pass the interface to the method integration code.

    ```python
    agent_model = agent_model_impl.build_model(device=device, config=config)
    ```

#### 3.1.2 Planner Model Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Detailed integration steps**:

  - **Step 1:** Create a planner adapter inherited from `PlannerModelInterface`.

    ```python
    from api.marl_method_interface import PlannerModelInterface

    class HAPPOPlannerModel(PlannerModelInterface):
        ...
    ```

  - **Step 2:** Implement planner construction, high-level planning, planner action/value inference, optimizer construction, and planner updates.

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

#### 3.1.3 Communication Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Detailed integration steps**:

  - **Step 1:** Create a communication adapter inherited from `CommunicationInterface`.

    ```python
    from api.marl_method_interface import CommunicationInterface

    class HAPPOCommunication(CommunicationInterface):
        ...
    ```

  - **Step 2:** Implement the HAPPO's message encoding, message decoding, and multi-agent message aggregation.

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

#### 3.1.4 Training Loop Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Detailed integration steps**:

  - **Step 1:** Create a training-loop adapter inherited from `TrainingLoopInterface`.

    ```python
    from api.marl_method_interface import TrainingLoopInterface

    class HAPPOTrainingLoop(TrainingLoopInterface):
        ...
    ```

  - **Step 2:** Implement environment construction, rollout collection, policy updates, evaluation, checkpointing, and the top-level run method.

    ```python
    training_loop = HAPPOTrainingLoop()
    results = training_loop.run(
        workload="object_stacking",
        config=config,
    )
    ```

  - **Step 3:** Connect the four interfaces in the training loop.

    ```python
    rollout = training_loop.collect_rollout(
        envs,
        agent_model,
        planner=planner,
        communication=communication,
        config=config,
    )
    ```

### 3.2 LLM-based MARL Example: ROCO<img src="./heading-divider.svg" alt="" width="100%" height="1">

You can add a new LLM-based MARL baseline according to the following steps.

#### 3.2.1 Agent Model Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Detailed integration steps**:

  - **Step 1:** Create a new file and define `ROCOAgentModel`, inherited from `AgentModelInterface`.

    ```python
    from api.marl_method_interface import AgentModelInterface

    class ROCOAgentModel(AgentModelInterface):
        ...
    ```

  - **Step 2:** Define the participating agents and their observation, language-context, and action dimensions for the selected workload.

    ```python
    model_impl = ROCOAgentModel(
        agent_names=("agent_0", "agent_1"),
        state_dims={"agent_0": 32, "agent_1": 32},
        action_dims={"agent_0": 7, "agent_1": 7},
    )
    ```

  - **Step 3:** Implement the LLM policy construction, conversion from observations and task context to a policy batch, action sampling, value estimation, and optimizer construction.

    ```python
    def build_model(self, *, device, config):
        # Return the ROCO language-conditioned actor-critic model.
        ...

    def build_batch_from_obs(self, obs, *, device):
        # Convert observations and task context to the model batch schema.
        ...

    def get_action_and_value(self, model, batch, *, actions_input=None, deterministic=False):
        # Return per-agent actions, log probabilities, entropies, and values.
        ...
    ```

  - **Step 4:** Initialize the agent model interface and construct the model.

    ```python
    agent_model_impl = ROCOAgentModel(
        agent_names=agent_names,
        state_dims=state_dims,
        action_dims=action_dims,
    )
    agent_model = agent_model_impl.build_model(device=device, config=config)
    ```

#### 3.2.2 Planner Model Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Detailed integration steps**:

  - **Step 1:** Create a planner adapter inherited from `PlannerModelInterface`.

    ```python
    from api.marl_method_interface import PlannerModelInterface

    class ROCOPlannerModel(PlannerModelInterface):
        ...
    ```

  - **Step 2:** Implement planner construction, generation of task-level decisions, planner action/value inference, optimizer construction, and planner updates.

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
    ```

#### 3.2.3 Communication Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Detailed integration steps**:

  - **Step 1:** Create a communication adapter inherited from `CommunicationInterface`.

    ```python
    from api.marl_method_interface import CommunicationInterface

    class ROCOCommunication(CommunicationInterface):
        ...
    ```

  - **Step 2:** Implement ROCO's message encoding, message decoding, and multi-agent message aggregation.

    ```python
    communication = ROCOCommunication()
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

#### 3.2.4 Training Loop Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Detailed integration steps**:

  - **Step 1:** Create a training-loop adapter inherited from `TrainingLoopInterface`.

    ```python
    from api.marl_method_interface import TrainingLoopInterface

    class ROCOTrainingLoop(TrainingLoopInterface):
        ...
    ```

  - **Step 2:** Implement environment construction, rollout collection, policy updates, evaluation, checkpointing, and the top-level run method.

    ```python
    training_loop = ROCOTrainingLoop()
    results = training_loop.run(
        workload="object_stacking",
        config=config,
    )
    ```

  - **Step 3:** Connect the ROCO interfaces in the training loop.

    ```python
    rollout = training_loop.collect_rollout(
        envs,
        agent_model,
        planner=planner,
        communication=communication,
        config=config,
    )
    ```

### 3.3 VLA-based MARL Example: MAPLE<img src="./heading-divider.svg" alt="" width="100%" height="1">

You can add a new VLA-based MARL baseline according to the following steps.

#### 3.3.1 Agent Model Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Detailed integration steps**:

  - **Step 1:** Create a new file and define `MAPLEAgentModel`, inherited from `AgentModelInterface`.

    ```python
    from api.marl_method_interface import AgentModelInterface

    class MAPLEAgentModel(AgentModelInterface):
        ...
    ```

  - **Step 2:** Define the VLA agent names and their state/action dimensions for the selected workload.

    ```python
    model_impl = MAPLEAgentModel(
        agent_names=("agent_0", "agent_1"),
        state_dims={"agent_0": 32, "agent_1": 32},
        action_dims={"agent_0": 7, "agent_1": 7},
    )
    ```

  - **Step 3:** Implement VLA model construction, conversion from visual-language observations to a policy batch, action sampling, value estimation, and optimizer construction.

    ```python
    def build_model(self, *, device, config):
        # Return the MAPLE VLA actor-critic model.
        ...

    def build_batch_from_obs(self, obs, *, device):
        # Convert visual observations, task text, and states to the policy batch schema.
        ...

    def get_action_and_value(self, model, batch, *, actions_input=None, deterministic=False):
        # Return per-agent actions, log probabilities, entropies, and values.
        ...
    ```

  - **Step 4:** Initialize the agent model interface and construct the VLA policy.

    ```python
    agent_model_impl = MAPLEAgentModel(
        agent_names=agent_names,
        state_dims=state_dims,
        action_dims=action_dims,
    )
    agent_model = agent_model_impl.build_model(device=device, config=config)
    ```

#### 3.3.2 Communication Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Detailed integration steps**:

  - **Step 1:** Create a communication adapter inherited from `CommunicationInterface`.

    ```python
    from api.marl_method_interface import CommunicationInterface

    class MAPLECommunication(CommunicationInterface):
        ...
    ```

  - **Step 2:** Implement MAPLE's message encoding, message decoding, and multi-agent message aggregation.

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

#### 3.3.3 Training Loop Interface<img src="./heading-divider-h4.svg" alt="" width="100%" height="1">

- **Detailed integration steps**:

  - **Step 1:** Create a training-loop adapter inherited from `TrainingLoopInterface`.

    ```python
    from api.marl_method_interface import TrainingLoopInterface

    class MAPLETrainingLoop(TrainingLoopInterface):
        ...
    ```

  - **Step 2:** Implement environment construction, rollout collection, MAPLE updates, evaluation, checkpointing, and the top-level run method.

    ```python
    training_loop = MAPLETrainingLoop()
    results = training_loop.run(
        workload="object_stacking",
        config=config,
    )
    ```

  - **Step 3:** Connect the MAPLE interfaces in the training loop.

    ```python
    rollout = training_loop.collect_rollout(
        envs,
        agent_model,
        communication=communication,
        config=config,
    )
    ```
