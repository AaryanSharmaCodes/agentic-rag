import math

def cosine_similarity(a: list[float],b: list[float]) -> float:
    dot = 0
    for x, y in zip(a,b):
        dot += x*y
    
    total=0
    for x in a:
        total += x**2
    mag_a = math.sqrt(total)
    
    total=0
    for x in b:
        total += x**2
    mag_b = math.sqrt(total)

    return dot/(mag_a * mag_b)

if __name__ == "__main__":
    print(cosine_similarity([1, 2, 2], [2, 4, 4]))   # expect 1.0
    print(cosine_similarity([1, 0], [0, 1]))          # expect 0.0
