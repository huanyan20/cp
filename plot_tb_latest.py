import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_latest_dir(parent_dir):
    all_subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return latest_subdir

tb_dir = "logs/tb_logs"
latest_dir = get_latest_dir(tb_dir)
print(f"Latest TensorBoard directory: {latest_dir}")

event_files = glob.glob(os.path.join(latest_dir, "events.out.tfevents.*"))
if not event_files:
    print("No event files found.")
    exit(1)

event_file = event_files[0]
ea = EventAccumulator(event_file)
ea.Reload()

tags = ea.Tags()['scalars']
print(f"Available scalar tags: {tags}")

plt.figure(figsize=(12, 5))
plot_idx = 1

# Plot reward if available
if 'rollout/ep_rew_mean' in tags:
    rew_events = ea.Scalars('rollout/ep_rew_mean')
    steps = [e.step for e in rew_events]
    vals = [e.value for e in rew_events]
    
    plt.subplot(1, 2, plot_idx)
    plt.plot(steps, vals, label='ep_rew_mean', color='blue')
    plt.title('Rollout Return (ep_rew_mean)')
    plt.xlabel('Timesteps')
    plt.ylabel('Return')
    plt.legend()
    plot_idx += 1

if 'train/loss' in tags or 'train/actor_loss' in tags:
    loss_tag = 'train/loss' if 'train/loss' in tags else 'train/actor_loss'
    loss_events = ea.Scalars(loss_tag)
    steps = [e.step for e in loss_events]
    vals = [e.value for e in loss_events]
    
    plt.subplot(1, 2, plot_idx)
    plt.plot(steps, vals, label=loss_tag, color='red')
    plt.title('Training Loss')
    plt.xlabel('Timesteps')
    plt.ylabel('Loss')
    plt.legend()

plt.tight_layout()
plt.savefig("tensorboard_learning_curve.png", dpi=150)
print("Saved learning curve to tensorboard_learning_curve.png")
