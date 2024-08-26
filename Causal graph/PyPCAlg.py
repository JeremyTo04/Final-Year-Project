from PyPCAlg.pc_algorithm import run_pc_algorithm, field_pc_cpdag, \
    field_separation_sets
from PyPCAlg.examples.graph_4 import generate_data
from PyPCAlg.examples.graph_4 import oracle_indep_test
from PyPCAlg.examples.graph_4 import oracle_cond_indep_test
from PyPCAlg.examples.graph_4 import get_adjacency_matrix


df = generate_data(sample_size=10)
independence_test_func = oracle_indep_test()
conditional_independence_test_func = oracle_cond_indep_test()

dic = run_pc_algorithm(
    data=df,
    indep_test_func=independence_test_func,
    cond_indep_test_func=conditional_independence_test_func,
    level=0.05
)
cpdag = dic[field_pc_cpdag]
separation_sets = dic[field_separation_sets]

print(f'The true causal graph is \n{get_adjacency_matrix()}')
print(f'\nThe CPDAG retrieved by PC is \n{cpdag}')