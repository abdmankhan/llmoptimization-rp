def pareto_front(solutions):
    pareto = []

    for s in solutions:
        dominated = False
        for o in solutions:
            if (
                o["cost"] <= s["cost"]
                and o["accuracy"] >= s["accuracy"]
                and (o["cost"] < s["cost"] or o["accuracy"] > s["accuracy"])
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(s)

    return pareto
