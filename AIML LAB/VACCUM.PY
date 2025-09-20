class VacuumCleaner:
    def __init__(self, shape):
        self.shape = shape
        self.state = 'rest'

    def command(self, cmd):
        if cmd == "start":
            self.state = 'cleaning'
            print(f"{self.shape} vacuum started cleaning.")
        elif cmd == "stop":
            self.state = 'rest'
            print(f"{self.shape} vacuum stopped.")
        elif cmd == "left":
            print(f"{self.shape} vacuum turned left.")
        elif cmd == "right":
            print(f"{self.shape} vacuum turned right.")
        elif cmd == "dock":
            self.state = 'docking'
            print(f"{self.shape} vacuum docking.")
        else:
            print(f"Command '{cmd}' not recognized.")



shapes = ["Circle", "Square", "Triangle", "Pentagon","Hexagon"]
print(shapes)
    
user_in=input("Enter the shape you want to clean from the shapes:")

if user_in.lower() in [s.lower() for s in shapes]:

    clean = [VacuumCleaner(user_in)]

    for cln in clean:
        cln.command("start")
        cln.command("left")
        cln.command("right")
        cln.command("dock")
        cln.command("stop")
        print()
    
else:
    print("Invalid shape please enter vaild shape from the given list")
    break
