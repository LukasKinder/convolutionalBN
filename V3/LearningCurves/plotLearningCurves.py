
import matplotlib.pyplot as plt

file_name = "firstExperiment"
#file_name = "niceShape"

data = []
proportions_white = []

with open(file_name, "r") as file:
    lines = file.readlines()
    for line in lines:
        split_line = line.split(" ")
        data.append( (float)(split_line[0]))

        for i in range(1,len(split_line)-1):
            if len(proportions_white) == 0:
                proportions_white = [[] for x in range(1,len(split_line))]
            
            proportions_white[i-1].append( (float)(split_line[i]))

plt.plot([x for x in data])
plt.xlabel("Iterations")
plt.ylabel("Entropy gain relations")
plt.grid()
plt.title("Entropy gain gradient descent")
plt.show()

for i in range(len(proportions_white)):
    plt.plot(proportions_white[i], label = "Kernel {0}".format(i))

plt.title("Proportions white")
plt.xlabel("Iterations")
plt.ylabel("proportions white")
plt.ylim((0.0,1.0))
plt.legend(fontsize=7)
plt.grid()
plt.show()


