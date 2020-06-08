import numpy as np
import math


class ProblemBase:
    def __init__(self):
        self.geneSize = 30
        self.variableLowerBound = 0
        self.variableUpperBound = 1
        self.x0LowerBound = 0
        self.x0UpperBound = 1
        self.objectives = [self.f1, self.f2]

    def g(self, gene):
        return 1 + (9 * np.sum(gene[1 : self.geneSize]) / (self.geneSize - 1))

    def f1(self, gene):
        return gene[0]

    def f2(self, gene):
        raise NotImplementedError

    def perfect_pareto_front(self):
        locations = np.empty([100, 2], dtype=float)
        locations[:, 0] = np.arange(0, 1, 0.01)

        locations[:, 1] = 1 - np.sqrt(locations[:, 0])

        return locations

    def createGene(self):
        return np.append(
            np.random.uniform(self.x0LowerBound, self.x0UpperBound, size=(1, 1)),
            np.random.uniform(
                self.variableLowerBound,
                self.variableUpperBound,
                size=(1, self.geneSize - 1),
            ),
        )

    def Mutate(self, gene):
        # Two random genes replaces with random numbers
        index1 = np.random.randint(0, self.geneSize)
        gene[index1] = np.random.uniform(
            self.variableLowerBound, self.variableUpperBound, size=(1)
        )
        return gene


class ZDT1(ProblemBase):
    def __init__(self):
        self.geneSize = 30
        self.variableLowerBound = 0
        self.variableUpperBound = 1
        self.x0LowerBound = 0
        self.x0UpperBound = 1
        self.objectives = [self.f1, self.f2]

    def f2(self, gene):
        h = 1 - math.sqrt(gene[0] / self.g(gene))
        return self.g(gene) * h


class ZDT2(ProblemBase):
    def __init__(self):
        self.geneSize = 30
        self.variableLowerBound = 0
        self.variableUpperBound = 1
        self.x0LowerBound = 0
        self.x0UpperBound = 1
        self.objectives = [self.f1, self.f2]

    def f2(self, gene):
        h = 1 - math.pow(gene[0] / self.g(gene), 2)
        return self.g(gene) * h

    def perfect_pareto_front(self):
        locations = np.empty([100, 2], dtype=float)
        locations[:, 0] = np.arange(0, 1, 0.01)

        locations[:, 1] = 1 - np.power(locations[:, 0], 2)

        return locations


class ZDT3(ProblemBase):
    def __init__(self):
        self.geneSize = 30
        self.variableLowerBound = 0
        self.variableUpperBound = 1
        self.x0LowerBound = 0
        self.x0UpperBound = 1
        self.objectives = [self.f1, self.f2]

    def f2(self, gene):
        h = (
            1
            - math.sqrt(gene[0] / self.g(gene))
            - (gene[0] / self.g(gene) * math.sin(10 * math.pi * gene[0]))
        )
        return self.g(gene) * h

    def perfect_pareto_front(self):
        locations = np.empty([100, 2], dtype=float)
        locations[:, 0] = np.arange(0, 1, 0.01)
        x1 = locations[:, 0]
        locations[:, 1] = (1 - np.sqrt(x1)) - x1 * np.sin(10 * math.pi * x1)

        return locations


class ZDT4(ProblemBase):
    def __init__(self):
        self.geneSize = 10
        self.variableLowerBound = -5
        self.variableUpperBound = 5

        self.x0LowerBound = 0
        self.x0UpperBound = 1
        self.objectives = [self.f1, self.f2]

    def g(self, gene):
        return (
            1
            + 10 * (self.geneSize - 1)
            + np.sum(
                np.power(gene[1 : self.geneSize], 2)
                - (10 * np.cos(4 * math.pi * gene[1 : self.geneSize]))
            )
        )

    def f2(self, gene):
        h = 1 - math.sqrt(gene[0] / self.g(gene))
        return self.g(gene) * h

    """def createGene(self):
        return np.append(
            np.random.uniform(self.x0LowerBound, self.x0UpperBound, size=(1, 1)),
            np.random.uniform(
                self.variableLowerBound,
                self.variableUpperBound,
                size=(1, self.geneSize - 1),
            ),
        )"""

    def Mutate(self, gene):
        # Two random genes replaces with random numbers
        number1 = np.random.randint(0, self.geneSize)
        number2 = np.random.randint(0, self.geneSize)

        if number1 == 0:
            gene[number1] = np.random.uniform(
                self.x0LowerBound, self.x0UpperBound, size=(1, 1)
            )
        else:
            gene[number1] = np.random.uniform(
                self.variableLowerBound, self.variableUpperBound, size=(1, 1)
            )

        if number2 == 0:
            gene[number1] = np.random.uniform(
                self.x0LowerBound, self.x0UpperBound, size=(1, 1)
            )
        else:
            gene[number1] = np.random.uniform(
                self.variableLowerBound, self.variableUpperBound, size=(1, 1)
            )

        return gene


# -----------------------------------UF-------------------------------------


class UF1(ProblemBase):
    def __init__(self):
        self.geneSize = 30
        self.variableLowerBound = -1
        self.variableUpperBound = 1
        self.x0LowerBound = 0
        self.x0UpperBound = 1
        self.objectives = [self.f1, self.f2]

    def commonPart(self, gene, objIndex):
        sum1 = sum2 = 0
        count1 = count2 = 0
        for j in range(2, self.geneSize):
            Yj = gene[j - 1] - math.sin(
                (6 * math.pi * gene[0]) + (j * math.pi / self.geneSize)
            )
            Yj = math.pow(Yj, 2)

            if j % 2 == 0:
                sum2 += Yj
                count2 += 1
            else:
                sum1 += Yj
                count1 += 1

        if objIndex == 0:
            return sum1, count1
        elif objIndex == 1:
            return sum2, count2

    def f1(self, gene):
        sum1, count1 = self.commonPart(gene, 0)
        return gene[0] + 2 * sum1 / count1

    def f2(self, gene):
        sum2, count2 = self.commonPart(gene, 1)
        return 1 - math.sqrt(gene[0]) + 2 * sum2 / count2

    def perfect_pareto_front(self):
        locations = np.empty([100, 2], dtype=float)
        locations[:, 0] = np.arange(0, 1, 0.01)
        locations[:, 1] = 1 - np.sqrt(locations[:, 0])
        return locations

    def perfect_pareto_set(self):
        locations = np.empty([100, 3], dtype=float)
        locations[:, 0] = np.arange(0, 1, 0.01)
        locations[:, 1] = np.sin(
            6 * math.pi * locations[:, 0] + 2 * math.pi / self.geneSize
        )
        locations[:, 2] = np.sin(
            6 * math.pi * locations[:, 0] + 3 * math.pi / self.geneSize
        )
        return locations


class UF2(ProblemBase):
    def __init__(self):
        self.geneSize = 30
        self.variableLowerBound = -1
        self.variableUpperBound = 1
        self.x0LowerBound = 0
        self.x0UpperBound = 1
        self.objectives = [self.f1, self.f2]

    def commonPart(self, gene, objIndex):
        sum1 = sum2 = 0
        count1 = count2 = 0
        for j in range(2, self.geneSize):
            if j % 2 == 0:
                Yj = gene[j - 1] - 0.3 * gene[0] * (
                    gene[0]
                    * math.cos(24 * math.pi * gene[0] + 4 * j * math.pi / self.geneSize)
                    + 2
                ) * math.sin(6 * math.pi * gene[0] + j * math.pi / self.geneSize)
                sum2 += math.pow(Yj, 2)
                count2 += 1
            else:
                Yj = gene[j - 1] - 0.3 * gene[0] * (
                    gene[0]
                    * math.cos(
                        24.0 * math.pi * gene[0] + 4.0 * j * math.pi / self.geneSize
                    )
                    + 2.0
                ) * math.cos(6.0 * math.pi * gene[0] + j * math.pi / self.geneSize)
                sum1 += math.pow(Yj, 2)
                count1 += 1

        if objIndex == 0:
            return sum1, count1
        elif objIndex == 1:
            return sum2, count2

    def f1(self, gene):
        sum1, count1 = self.commonPart(gene, 0)
        return gene[0] + 2 * sum1 / count1

    def f2(self, gene):
        sum2, count2 = self.commonPart(gene, 1)
        return 1 - math.sqrt(gene[0]) + 2 * sum2 / count2

    def perfect_pareto_front(self):
        locations = np.empty([100, 2], dtype=float)
        locations[:, 0] = np.arange(0, 1, 0.01)
        locations[:, 1] = 1 - np.sqrt(locations[:, 0])
        return locations

    def perfect_pareto_set(self):
        locations = np.empty([100, 3], dtype=float)
        locations[:, 0] = np.arange(0, 1, 0.01)
        locations[:, 1] = (
            0.3
            * np.power(locations[:, 0], 2)
            * np.cos(24 * math.pi * locations[:, 0] + 4 * 2 * math.pi / self.geneSize)
            + 0.6 * locations[:, 0]
        ) * np.sin(6 * math.pi * locations[:, 0] + 2 * math.pi / self.geneSize)
        locations[:, 2] = (
            0.3
            * np.power(locations[:, 0], 2)
            * np.cos(24 * math.pi * locations[:, 0] + 4 * 3 * math.pi / self.geneSize)
            + 0.6 * locations[:, 0]
        ) * np.cos(6 * math.pi * locations[:, 0] + 3 * math.pi / self.geneSize)
        return locations


class UF3(ProblemBase):
    def __init__(self):
        self.geneSize = 30
        self.variableLowerBound = -1
        self.variableUpperBound = 1
        self.x0LowerBound = 0
        self.x0UpperBound = 1
        self.objectives = [self.f1, self.f2]

    def commonPart(self, gene, objIndex):
        sum1 = sum2 = 0
        count1 = count2 = 0
        prod1 = prod2 = 1
        for j in range(2, self.geneSize):
            Yj = gene[j - 1] - math.pow(
                gene[0], 0.5 * (1.0 + 3.0 * (j - 2.0) / (self.geneSize - 2.0))
            )
            Pj = math.cos(20.0 * Yj * math.pi / math.sqrt(j))
            if j % 2 == 0:
                sum2 += math.pow(Yj, 2)
                prod2 *= Pj
                count2 += 1
            else:
                sum1 += math.pow(Yj, 2)
                prod1 *= Pj
                count1 += 1

        if objIndex == 0:
            return sum1, count1, prod1
        elif objIndex == 1:
            return sum2, count2, prod2

    def f1(self, gene):
        sum1, count1, prod1 = self.commonPart(gene, 0)
        return gene[0] + 2 * (4 * sum1 - 2 * prod1 + 2) / count1

    def f2(self, gene):
        sum2, count2, prod2 = self.commonPart(gene, 1)
        return 1 - math.sqrt(gene[0]) + 2 * (4 * sum2 - 2 * prod2 + 2) / count2

    def perfect_pareto_front(self):
        locations = np.empty([100, 2], dtype=float)
        locations[:, 0] = np.arange(0, 1, 0.01)
        locations[:, 1] = 1 - np.sqrt(locations[:, 0])
        return locations

    def perfect_pareto_set(self):
        locations = np.empty([100, 3], dtype=float)
        locations[:, 0] = np.arange(0, 1, 0.01)
        locations[:, 1] = np.power(
            locations[:, 0], 0.5 * (1 + (3 * (2 - 2)) / (self.geneSize - 2))
        )
        locations[:, 2] = np.power(
            locations[:, 0], 0.5 * (1 + (3 * (3 - 2)) / (self.geneSize - 2))
        )
        return locations


class UF4(ProblemBase):
    def __init__(self):
        self.geneSize = 30
        self.variableLowerBound = -2
        self.variableUpperBound = 2
        self.x0LowerBound = 0
        self.x0UpperBound = 1
        self.objectives = [self.f1, self.f2]

    def commonPart(self, gene, objIndex):
        sum1 = sum2 = 0
        count1 = count2 = 0
        for j in range(2, self.geneSize):
            Yj = gene[j - 1] - math.sin(
                6.0 * math.pi * gene[0] + j * math.pi / self.geneSize
            )
            Hj = abs(Yj) / (1.0 + math.exp(2.0 * abs(Yj)))
            if j % 2 == 0:
                sum2 += Hj
                count2 += 1
            else:
                sum1 += Hj
                count1 += 1

        if objIndex == 0:
            return sum1, count1
        elif objIndex == 1:
            return sum2, count2

    def f1(self, gene):
        sum1, count1 = self.commonPart(gene, 0)
        return gene[0] + 2 * sum1 / count1

    def f2(self, gene):
        sum2, count2 = self.commonPart(gene, 1)
        return 1 - math.pow(gene[0], 2) + 2 * sum2 / count2

    def perfect_pareto_front(self):
        locations = np.empty([100, 2], dtype=float)
        locations[:, 0] = np.arange(0, 1, 0.01)
        locations[:, 1] = 1 - np.power(locations[:, 0], 2)
        return locations

    def perfect_pareto_set(self):
        locations = np.empty([100, 3], dtype=float)
        locations[:, 0] = np.arange(0, 1, 0.01)
        locations[:, 1] = np.sin(
            6.0 * math.pi * locations[:, 0] + 2 * math.pi / self.geneSize
        )
        locations[:, 2] = np.sin(
            6.0 * math.pi * locations[:, 0] + 3 * math.pi / self.geneSize
        )
        return locations


class UF5(ProblemBase):
    def __init__(self):
        self.geneSize = 30
        self.N = 10
        self.E = 0.1
        self.variableLowerBound = -1
        self.variableUpperBound = 1
        self.x0LowerBound = 0
        self.x0UpperBound = 1
        self.objectives = [self.f1, self.f2]

    def commonPart(self, gene, objIndex):
        sum1 = sum2 = 0
        count1 = count2 = 0

        for j in range(2, self.geneSize):
            Yj = gene[j - 1] - math.sin(
                6.0 * math.pi * gene[0] + j * math.pi / self.geneSize
            )
            Hj = 2.0 * math.pow(Yj, 2) - math.cos(4.0 * math.pi * Yj) + 1.0
            if j % 2 == 0:
                sum2 += Hj
                count2 += 1
            else:
                sum1 += Hj
                count1 += 1
        Hj = (0.5 / self.N + self.E) * abs(math.sin(2.0 * self.N * math.pi * gene[0]))
        if objIndex == 0:
            return sum1, count1, Hj
        elif objIndex == 1:
            return sum2, count2, Hj

    def f1(self, gene):
        sum1, count1, Hj = self.commonPart(gene, 0)
        return gene[0] + Hj + 2 * sum1 / count1

    def f2(self, gene):
        sum2, count2, Hj = self.commonPart(gene, 1)
        return 1 - gene[0] + Hj + 2 * sum2 / count2

    def perfect_pareto_front(self):
        locations = np.empty([2 * self.N + 1, 2], dtype=float)
        locations[:, 0] = np.arange(0, 1.05, 0.05)  # (i/2N,
        locations[:, 1] = 1 - locations[:, 0]  # 1 −i/2N)
        return locations


class UF6(ProblemBase):
    def __init__(self):
        self.geneSize = 30
        self.N = 2
        self.E = 0.1
        self.variableLowerBound = -1
        self.variableUpperBound = 1
        self.x0LowerBound = 0
        self.x0UpperBound = 1
        self.objectives = [self.f1, self.f2]

    def commonPart(self, gene, objIndex):
        sum1 = sum2 = 0
        count1 = count2 = 0
        prod1 = prod2 = 1
        for j in range(2, self.geneSize):
            Yj = gene[j - 1] - math.sin(
                6.0 * math.pi * gene[0] + j * math.pi / self.geneSize
            )
            Pj = math.cos(20 * Yj * math.pi / math.sqrt(j))
            if j % 2 == 0:
                sum2 += Yj * Yj
                prod2 *= Pj
                count2 += 1
            else:
                sum1 += Yj * Yj
                prod1 *= Pj
                count1 += 1
        Hj = 2 * (0.5 / self.N + self.E) * math.sin(2.0 * self.N * math.pi * gene[0])
        if objIndex == 0:
            return sum1, count1, Hj, prod1
        elif objIndex == 1:
            return sum2, count2, Hj, prod2

    def f1(self, gene):
        sum1, count1, Hj, prod1 = self.commonPart(gene, 0)
        return gene[0] + Hj + 2 * (4 * sum1 - 2 * prod1 + 2) / count1

    def f2(self, gene):
        sum2, count2, Hj, prod2 = self.commonPart(gene, 1)
        return 1 - gene[0] + Hj + 2 * (4 * sum2 - 2 * prod2 + 2) / count2

    def perfect_pareto_front(self):
        locations = np.empty([51, 2], dtype=float)
        # one isolated point, (0, 1)
        locations[50, 0] = 0
        locations[50, 1] = 1

        locations[:25, 0] = np.arange(0.75, 1, 0.01)
        locations[25:50, 0] = np.arange(0.25, 0.5, 0.01)
        locations[:50, 1] = 1 - locations[:50, 0]
        return locations


class UF7(ProblemBase):
    def __init__(self):
        self.geneSize = 30
        self.variableLowerBound = -1
        self.variableUpperBound = 1
        self.x0LowerBound = 0
        self.x0UpperBound = 1
        self.objectives = [self.f1, self.f2]

    def commonPart(self, gene, objIndex):
        sum1 = sum2 = 0
        count1 = count2 = 0
        for j in range(2, self.geneSize):
            Yj = gene[j - 1] - math.sin(
                6.0 * math.pi * gene[0] + j * math.pi / self.geneSize
            )
            if j % 2 == 0:
                sum2 += Yj * Yj
                count2 += 1
            else:
                sum1 += Yj * Yj
                count1 += 1

        if objIndex == 0:
            return sum1, count1
        elif objIndex == 1:
            return sum2, count2

    def f1(self, gene):
        sum1, count1 = self.commonPart(gene, 0)
        return math.pow(gene[0], .2) + 2 * sum1 / count1

    def f2(self, gene):
        sum2, count2 = self.commonPart(gene, 1)
        return 1 - math.pow(gene[0], .2) + 2 * sum2 / count2

    def perfect_pareto_front(self):
        locations = np.empty([100, 2], dtype=float)
        locations[:, 0] = np.arange(0, 1, 0.01)
        locations[:, 1] = 1 - locations[:, 0]
        return locations

    def perfect_pareto_set(self):
        locations = np.empty([100, 3], dtype=float)
        locations[:, 0] = np.arange(0, 1, 0.01)
        locations[:, 1] = np.sin(
            6.0 * math.pi * locations[:, 0] + 2 * math.pi / self.geneSize
        )
        locations[:, 2] = np.sin(
            6.0 * math.pi * locations[:, 0] + 3 * math.pi / self.geneSize
        )
        return locations

