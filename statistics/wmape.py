forecast = [100.,200.,300.]
actual = [50.,210.,275.]

def wmape(f,a):
    x = 0.0
    for i,j in zip(f,a):
        x += abs(i-j)/j
    return (1 - x/len(f)) * 100


print wmape(forecast,actual)

