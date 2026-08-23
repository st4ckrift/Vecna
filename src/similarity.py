def euclidean_dis(a,b):
    lis=list(zip(a.values,b.values))
    total=0
    if a.values.len() != b.values.len():
        return 
    else:
        for ele in lis:
            total+=((ele[1]-ele[0])**2)
    return(total**0.5)

