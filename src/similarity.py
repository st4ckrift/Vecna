def euclidean_dis(a,b):
    pair=list(zip(a.values,b.values))
    total=0
    if len(a.values) != len(b.values):
        raise ValueError("Length of both lists should be same")
    for ele in pair:
        total+=((ele[1]-ele[0])**2)
    return(total**0.5)

