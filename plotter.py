from pandas import read_csv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = read_csv('dynaq.csv')
    df2 = read_csv('dynaqplus.csv')
    dynq = df.plot(x='episode', y='steps', label='dynaq')
    dynqp = df2.plot(x='episode', y='steps', label='dynaq+', ax=dynq)
    dynqp.set_xlabel("episodes")
    dynqp.set_ylabel("steps")
    dynqp.set_title(f"DynaQ vs DynaQ+\nAlpha: 1.0 Gamma: 0.95 Épsilon: 0.1 Kappa: 0.0001")
    plt.show()
