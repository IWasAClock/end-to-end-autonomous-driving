import carla
import torch
import torch.nn as nn
from collections import deque
from typing import List

class CarlaDataCollector:
    """Collects training data in CARLA using autopilot."""

    def __init__(self, host: str = "localhost", port: int = 2000, frame_stack: int = 1):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def collect(self, duration_seconds: int = 30) -> List[torch.Tensor]:
        """Collect images and control commands for a fixed duration."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])
        vehicle.set_autopilot(True)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera = self.world.spawn_actor(camera_bp, carla.Transform(), attach_to=vehicle)
        images, controls = [], []

        def _on_image(image):
            array = image.raw_data
            tensor = torch.as_tensor(array, dtype=torch.uint8)
            images.append(tensor)
            control = vehicle.get_control()
            controls.append((control.steer, control.throttle, control.brake))

        camera.listen(_on_image)
        self.world.wait_for_tick()
        for _ in range(int(duration_seconds * 20)):
            self.world.tick()
        camera.stop()
        vehicle.destroy()
        camera.destroy()
        return images, controls


class SimpleCNN(nn.Module):
    """A tiny convolutional network mapping images to control commands."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 36 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(images: List[torch.Tensor], controls: List[tuple], epochs: int = 5) -> SimpleCNN:
    """Train the network with collected data."""
    # TODO: implement batching and normalization
    model = SimpleCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        for img, ctrl in zip(images, controls):
            img = img.float().permute(2, 0, 1) / 255.0
            target = torch.tensor(ctrl, dtype=torch.float32)
            pred = model(img.unsqueeze(0))
            loss = criterion(pred.squeeze(0), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def run_agent(model: SimpleCNN, host: str = "localhost", port: int = 2000):
    """Run the trained agent in CARLA."""
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera = world.spawn_actor(camera_bp, carla.Transform(), attach_to=vehicle)

    def _on_image(image):
        array = image.raw_data
        img = torch.as_tensor(array, dtype=torch.float32).permute(2, 0, 1) / 255.0
        with torch.no_grad():
            steer, throttle, brake = model(img.unsqueeze(0)).squeeze(0)
        vehicle.apply_control(carla.VehicleControl(
            steer=float(steer), throttle=float(throttle), brake=float(brake)
        ))

    camera.listen(_on_image)
    while True:
        world.tick()

