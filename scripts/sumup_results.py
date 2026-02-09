from statistics import mean

def find_lines_with_key(filename, key):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        matching_lines = [line.strip() for line in lines if key in line]
        return matching_lines

global_opt = 567.312555400384

print("\n%25s %25s %25s %25s %25s seconds %25s %25s" %('constraints', 'target', 'BO (1 step)', 'MAPR [%]', 'avg_exec_time', 'avg_single_points', 'avg_perc_repeated [%]'))

for constr in ['none', 'local', 'global']:
    for target in ['none', 'local', 'global']:
        for BO in ['none', 'BO']:
            filename = f'output/output_{constr}_{target}_{BO}.txt'

            # MAPR
            key = 'Best Fitness Value ABC'
            lines = find_lines_with_key(filename, key)
            values = [float(lines[_].split("Value ABC: ")[1].split(", MAPR")[0]) for _ in range(len(lines)-1)]
            MAPR = round((mean(values) - global_opt)/global_opt * 100, 2)

            # Execution Time
            key = 'Exectuion time'
            lines = find_lines_with_key(filename, key)
            values = [float(lines[_].split("Exectuion time: ")[1].split(" seconds")[0]) for _ in range(len(lines))]
            avg_exec_time = round(mean(values), 2)

            # Single points evaluated
            key = 'Number of non-repeated points:'
            lines = find_lines_with_key(filename, key)
            values = [int(lines[_].split("non-repeated points: ")[1]) for _ in range(len(lines))]
            avg_single_points = int(mean(values))

            # Repeated points
            key = 'of the evaluated points are repeated'
            lines = find_lines_with_key(filename, key)
            values = [float(lines[_].split("%")[0]) for _ in range(len(lines))]
            avg_perc_repeated = round(mean(values), 2)

            print("%25s %25s %25s %25s %25s seconds %25s %25s" %(constr, target, BO, MAPR, avg_exec_time, avg_single_points, avg_perc_repeated))


            

            





