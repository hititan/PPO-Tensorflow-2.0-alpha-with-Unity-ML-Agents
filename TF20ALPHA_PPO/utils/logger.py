color2code= dict(
    warning='\033[1;41m',
    ok='\033[1;32;40m',
    info='\033[1;36;40m'
)

def log(str="", color="info"):

    col = color2code[color]
    end ="\033[0m" + "\n"

    print("\n" + col + "---------------------------------")
    print(str)
    print("---------------------------------" + end)
