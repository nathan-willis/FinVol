def create_batches(minx,dx,maxx,miny,dy,maxy):
    x = minx
    L = []
    while x<=maxx+1e-10:
        y = miny
        while y<=maxy+1e-10:
            if x*y<=1+1e-8: L.append([x,y])
            y+=dy
        x+=dx

    R = list(range(0,len(L),int(len(L)/(int(len(L)/256)+1))+1)) + [len(L)]
    for i in range(len(R)-1):
        l = L[R[i]:R[i+1]]
        with open('./batch_%i.txt'%i,'w') as f:
            for params in l:
                f.write('%0.2f %0.2f\n'%(params[0],params[1]))
