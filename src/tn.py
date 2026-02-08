from sys import argv

def temp_demand(x: float):
    return round( 200 - 15*x + 0.8*(x**2) - 0.01*(x**3), 3)

if __name__ == "__main__":
    temp = float(argv[1])
    print(f"t({temp}) = {temp_demand(temp)}")