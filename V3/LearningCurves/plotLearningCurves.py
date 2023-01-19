
import matplotlib.pyplot as plt

file_name = "firstExperiment"

data = []
proportion_white = []

with open(file_name, "r") as file:
    lines = file.readlines()
    for line in lines:
        split_line = line.split(" ")
        data.append( (float)(split_line[0]))
        proportion_white.append( (float)(split_line[1]))

plt.plot([x*10 for x in data], label = "heuristic")
plt.xlabel("Iterations")
plt.ylabel(" A(P(N|Rel.)) - Av(P(N|~Rel.))")

plt.plot(proportion_white, label = "proprtion_white")
plt.legend()
plt.show()


