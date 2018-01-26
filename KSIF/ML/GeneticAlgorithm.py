"""
Genetic Algorithm module

 이 코드는 Genetic Algorithm으로 우월한 전략을 확률적으로 찾기위해 만들었습니다.
 Genetic Algorithm은 다음과 같이 구성하였습니다.

 1. 먼저 랜덤하게 population을 생성하고,
 2. 그 중에서 우수한 전략만 선택합니다.
 3. 선택한 전략들을 교배(CrossOver), 혹은 변이(Mutation)시켜 새로운 전략을 만듭니다.
 4. 추가로 랜덤하게 새로운 전략을 population에 추가시켜 Next Generation을 만듭니다.
 5. 1~4 를 반복시켜 전략 집합을 진화(Evolution)시킵니다.

 이를 통해 최종적으로 우수한 전략들이 얻어지게 됩니다.

 (Richard Tymerski et al, Proceedings of the Second Australasian Conference
  on Artificial Life and Computational Intelligence - Volume 9592, 2016)

"""
__author__ = 'Seung Hyeon Yu'
__email__  = 'rambor12@business.kaist.ac.kr'

# TODO : 손봐야함.
import functools
import random
from KSIF.core import base, io
from KSIF.util import format
import deap
from deap import creator, tools

NPOP = 60  # 총 인구
NSUR = 20  # 다음 세대까지 살아남을 인구 ( 2명씩 짝지어서 자식낳아야하므로 꼭 2의 배수여야함)
CXPB = 0.5  # CrossOver 할 확률
MUTPB = 0.3  # Mutation할 확률
NGEN = 5  # 세대 수
NCYCLE = 10  # Evolution Cycle 수


def prettyprint(pop):
    """
     population을 예쁘게 Print
    :param pop: population
    """
    for key in pop[0].keys():
        print("\t ", key, end="")
    print("\t\t Avg Return")
    i = 1
    for ind in sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True):
        print(str(i), "\t", list(map(format.prettyfloat, ind.values())), "\t",
              format.prettyfloat(ind.fitness.values[0] * 100), " %")
        i += 1


def initRandom(strategy):
    """
     초기에 Strategy 만드는 function
    :param strategy: Strategy Type
    :return: Initial Strategy
    """
    init = base.StrategyBasic({'PERr': random.uniform(0.33, 1),
                     'PBRr': random.uniform(0.33, 1),
                     'PCRr': random.uniform(0.33, 1)})
    init.part = 0.33
    return init


def high_select(pop, select_num):
    """
    선택하는 기준 : 보통은 performance measure 랭킹 세워서 순서대로 자름
    :param pop: population
    :param select_num: 골라낼 individual 수
    :return: 골라낸 population
    """
    sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True)
    return [ind for idx, ind in enumerate(sorted_pop) if idx < select_num]


def twoPointCrossover(ind1, ind2, cross_prob=0.5):
    """
     individual 1과 individual 2를 dictionary data 를 섞음
    :param cross_prob: 섞일 확률
    :return: Cross over 된 individual
    """
    for (key1, value1), (key2, value2) in zip(list(ind1.items()), list(ind2.items())):
        if random.random() < cross_prob:
            ind1[key1] = value2
            ind2[key2] = value1
        else:
            ind1[key1] = value1
            ind2[key2] = value2
    return (ind1, ind2)


def mutGaussian(ind, mu=0, sigma=0.1):
    """
     Gaussian 분포로 변이
    :param ind: 개인
    :param mu: 평균
    :param sigma: 표준편차
    :return: 변이된 개인
    """
    for key, value in ind.items():
        rand_num = random.gauss(value + mu, sigma)
        while (not rand_num > ind.part) or (not rand_num < 1):
            rand_num = random.gauss(value + mu, sigma)
        ind[key] = rand_num
    return (ind)


def avg_ret_evaluate(data, individual):
    # Build your own portfolio
    port_eval = individual.buildport(data)
    # Set your own criteria like Sharpe Ratio
    avg_retm = base.backtest(port_eval)['RETM'].mean()
    return (avg_retm,)


class Evolution:
    """
     Genetic algorithm을 이용하여 진화시키는 class
     객체를 생성하여 run 메서드를 실행시키고 결과를 print할 수 있다.
    """
    def __init__(self,
                 data,
                 strategy=base.Strategy,
                 init_strategies=None,
                 initialization=initRandom,
                 evaluate=avg_ret_evaluate,
                 mate=twoPointCrossover,
                 mutate=mutGaussian,
                 select=high_select,
                 NPOP = 60,
                 NSUR = 20,
                 CXPB = 0.5,
                 MUTPB = 0.3,
                 NGEN = 5,
                 NCYCLE = 10
                 ):
        """
        초기화
        :param strategy: 전략 클래스
        :param init_strategies: 초기 population으로 사용할 전략 list
        :param initialization: 전략 클래스를 input으로 받고 initialize된 전략을 return하는 함수
        :param evaluate: input = (data, strategy), out = performance 인 함수
        :param mate:
        :param mutate:
        :param select:
        :param NPOP:
        :param NSUR:
        :param CXPB:
        :param MUTPB:
        :param NGEN:
        :param NCYCLE:
        """
        self.data = data
        self.strategy = strategy({'PERr': 1})
        self.init_strategies = init_strategies
        self.initialization = initialization
        self.evaluate = evaluate
        self.mate = mate
        self.mutate = mutate
        self.select = select
        self.NPOP = NPOP
        self.NSUR = NSUR
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.NGEN = NGEN
        self.NCYCLE = NCYCLE
        self.log = " #\tStrategy Info\tPerformance\t\n"

    @property
    def run(self):
        # 1. Type Creating
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", self.strategy, fitness=creator.FitnessMax)

        # 2. Initialization : create individual, create population
        toolbox = deap.base.Toolbox()
        toolbox.register("individual", self.initialization, creator.Individual)
        # TODO: strategy list를 넣을 수 있도록!
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def clone(ind):
            copy = toolbox.individual()
            for k, v in ind.items():
                copy[k] = v
            copy.fitness.values = ind.fitness.values
            return copy

        # 3. Operators : Register Genetic Operators
        toolbox.register("mate", self.mate)
        toolbox.register("mutate", self.mutate)
        toolbox.register("select", self.select)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("clone", clone)

        # 4. Go
        bestavg = self.initialization(self.strategy)
        for key in bestavg.keys():
            bestavg[key] = 0
        bestlist = []

        for i in range(NCYCLE):
            pop = toolbox.population(n=NPOP)
            # Evaluate the entire population
            fitnesses = map(functools.partial(toolbox.evaluate, self.data), pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            for g in range(NGEN):
                result = "\n ----------------- GENERATION : "+str(g)+"-----------------"
                print(result)
                self.log.append(result)
                prettyprint(pop)
                survived_pop = toolbox.select(pop, self.NSUR)
                offspring = list(map(toolbox.clone, survived_pop))
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.CXPB:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in survived_pop:
                    if random.random() < self.MUTPB:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                new_pop = toolbox.population(n=self.NPOP - 2 * self.NSUR)
                invalid_ind = [ind for ind in offspring + survived_pop + new_pop if not ind.fitness.valid]
                fitnesses = list(map(functools.partial(toolbox.evaluate, self.data), invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                pop[:] = survived_pop + offspring + new_pop

            result = "\n ----------------- GENERATION : FINAL -----------------"
            print(result)
            self.log.append(result)
            prettyprint(pop)
            sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True)
            print(" BEST STRATEGY is ...")
            prettyprint([sorted_pop[0]])
            for key, value in sorted_pop[0].items():
                bestavg[key] += value
            bestlist.append(sorted_pop[0])

        print("--------------------- RESULT -------------------")
        prettyprint(bestlist)
        print(" AVG BEST STRATEGY is ...")
        for key, value in bestavg.items():
            bestavg[key] = bestavg[key] / 10
            print(key, " : ", round(bestavg[key], 2), " %")

        return bestavg.buildport(self.data)


def _test():
    data = io.read('\\..\\tests\\data\\input')
    mask = (data.FNSECTCODE != "FGSC.40") & (~ data.FNSECTCODE.isnull())
    data = data[mask]
    original_list = ['MVmr', 'PERr', 'PBRr', 'PCRr']
    data = io.cleanse(data, selector=base.Strategy({key: 1 for key in original_list}))
    data = data[data.MVmr < 0.5]
    port = Evolution(data)
    port_retm = base.backtest(port, '2011-01-01', '2016-01-02')
    print("Avg Ret : ", round(port_retm['RETM'].mean() * 100, 2), " %")
    io.graph(port_retm, label="최종진화전략")


if __name__ == '__main__':
    _test()
