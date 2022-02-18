import pickle
import os 
import skimage.io as io

d = {"a": 1, "b": 2, "afaefa": 5}
f = 2

with open(r"test.p", "wb") as output_file:
    pickle.dump(d, output_file,-1)
    pickle.dump(f, output_file, -1)

with open(r"test.p", "rb") as input_file:
    e = pickle.load(input_file)
    g = pickle.load(input_file)


print(e)
print(g)

load_path = "rl_data/saves_1"
nb_files = int(len([f for f in os.listdir(load_path) if f.endswith('.p') and os.path.isfile(os.path.join(load_path, f))]) / 5) # five dicts
for i in range(nb_files):
    if i == 148:
        continue # missing depth image 148 (copy error)
    
    obs_load          = pickle.load(open(load_path + "/obs_dump" +str(i) + ".p", "rb"))
    di_load           = pickle.load(open(load_path + "/di_dump" + str(i) + ".p", "rb"))
    action_load       = pickle.load(open(load_path + "/action_dump" + str(i) + ".p", "rb"))
    action_index_load = pickle.load(open(load_path + "/action_index_dump" + str(i) + ".p", "rb"))
    collision_load    = pickle.load(open(load_path + "/collision_dump" + str(i) + ".p", "rb"))

    print(i)