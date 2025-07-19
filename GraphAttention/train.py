from run import run_training,load_snapshots_from_dump
print("training started")
snapshots = load_snapshots_from_dump("out_small.dump")  # path to your LAMMPS dump
model = run_training(snapshots, epochs=1000, batch_size=10)
print("Training Ended")

