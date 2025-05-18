# Example: End-to-End Driving Pipeline in CARLA

This folder contains a minimal example of how one might implement an end-to-end
autonomous driving loop using the [CARLA](https://carla.org) simulator. The code
is intentionally simple and focuses on illustrating the workflow:

1. **Data Collection** – Run the CARLA autopilot to record camera images and
the corresponding control commands.
2. **Model Training** – Train a small convolutional neural network to imitate the
collected behavior.
3. **Agent Execution** – Deploy the trained model back into CARLA for closed-loop
evaluation.

Run the following commands from this directory after starting the CARLA server:

```bash
python carla_e2e_pipeline.py
```

The script assumes the CARLA Python API is installed and that a simulator is
running on `localhost:2000`.

*Note:* The provided implementation is simplified and meant for demonstration
purposes. Real-world autonomous driving requires additional components such as
sensor fusion, data augmentation, and safety checks.
