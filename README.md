# Gorchakov-BS-thesis
Репозиторий бакалаврского диплома Горчакова В.В.
# Importance Sampling Approach to Chance-Constrained DC Optimal Power Flow

# Научный руководитель: Максимов Ю.В.

#Short description

Despite significant economic and ecological effects, a higher level of renewable energy generation leads to increased uncertainty and variability in power injections, thus compromising grid reliability. In order to improve power grid security, we investigate a joint chance-constrained (CC) direct current (DC) optimal power flow (OPF) problem. The problem aims to find economically optimal power generation while guaranteeing that all power generation, line flows, and voltages simultaneously remain within their bounds with a pre-defined probability. Unfortunately, the problem is computationally intractable even if the distribution of renewables fluctuations is specified. Moreover, existing approximate solutions to the joint CC OPF problem are overly conservative, and therefore have less value for the operational practice. This paper proposes an importance sampling approach to the CC DC OPF problem, which yields better complexity and accuracy than current state-of-the-art methods. The algorithm efficiently reduces the number of scenarios by generating and using only the most important of them, thus enabling real-time solutions for test cases with up to several hundred buses.

Requirements
[Ссылка](https://github.com/Intelligent-Systems-Phystech/Gorchakov-BS-thesis/blob/master/code/requirements.txt)

Code
[Ссылка](https://github.com/Intelligent-Systems-Phystech/Gorchakov-BS-thesis/blob/master/code/constr_formulations.py)

Demo
[Ссылка](https://github.com/Intelligent-Systems-Phystech/Gorchakov-BS-thesis/blob/master/code/plots_for_paper.ipynb)

