#!/usr/bin/python3
# encoding: utf-8

import numpy as np
from sympy import Symbol, Function
import sys
import matplotlib.pyplot as plt

from sympy import diff
import sympy as sym

from sympy.abc import x, y, n

class Problem():

    x0 = 0
    xf = 0
    npoint = 0
    f = [0,0,0]
    f_analytic = 0

    def __init__(self, x0, xf, npoint, f, f_analytic):
        from sympy.abc import x, y, n

        self.x0 = x0
        self.xf = xf
        self.npoint = npoint
        self.f1 = f[0]
        self.f2 = f[1]
        self.f3 = f[2]
        self.f_analytic = f_analytic

    def collocation_method(self):

        # pontos igualmente espaçados #
        dx = (self.xf - self.x0)/(self.npoint + 1)
        point = np.array([i for i in np.linspace(self.x0+dx, self.xf-dx, self.npoint ) ])        
        from sympy.abc import x, y, n

        a = sym.IndexedBase('a')
        u1 = sym.summation(a[n] * eval(self.f1), (n, 1, len(point)))
        u2 = sym.summation(a[n] * eval(self.f2), (n, 1, len(point)))
        u3 = sym.summation(a[n] * eval(self.f3), (n, 1, len(point)))

        r1 = sym.diff(u1, x, x) + u1 - 2*x
        r2 = sym.diff(u2, x, x) + u2 - 2*x
        r3 = sym.diff(u3, x, x) + u3 - 2*x

        R1 = [r1.subs(x, coord) for coord in point]
        R2 = [r2.subs(x, coord) for coord in point]
        R3 = [r3.subs(x, coord) for coord in point]
       
        out = open("collocation_comparation_n="+str(self.npoint)+".txt", "w")  # open output file
        out.write(f'Intervalo de {self.x0} a {self.xf} dividido em {self.npoint} pontos.\n\n')
        out.write("   x      a[n]        f1            f2           f3              Exata\n")

        sol1 = sym.solve(R1, dict=True)
        # print(sol1)
        sol2 = sym.solve(R2, dict=True)
        # print(sol2)
        sol3 = sym.solve(R3, dict=True)
        # print(sol3)

        sol_exact = eval(f_analytic)

        for var in sol1[0]: u1 = u1.subs(var, sol1[0][var])
        for var in sol2[0]: u2 = u2.subs(var, sol2[0][var])
        for var in sol3[0]: u3 = u3.subs(var, sol3[0][var])


        Ye =  [sol_exact.subs(x,coord) for coord in point]
        Y1  = [u1.subs(x,coord) for coord in point]
        Y2  = [u2.subs(x,coord) for coord in point]
        Y3  = [u3.subs(x,coord) for coord in point]



        for n in range(0, len(point)):
            out.write(f' {point[n]:.3f}    {a[n]}    {str(Y1[n]).rjust(5)[:10]}    {str(Y2[n]).rjust(1)[:10]}    {str(Y3[n]).rjust(1)[:10]}   {sol_exact.subs(x,point[n])} \n')
        out.write('\n \n')
        out.write('   x      a[n]              f1(Erro)\n')

        for i in range(0, len(point)):
            out.write(f' {point[i]:.3f}    {a[i]}        {abs(Y1[i] - (sol_exact.subs(x,point[i]))) ** 2} \n')
        out.write('----'*25)
        out.write('\n')
        out.write('   x      a[n]              f2(Erro)\n')

        for i in range(0, len(point)):
            out.write(f' {point[i]:.3f}    {a[i]}        {abs(Y2[i] - (sol_exact.subs(x,point[i]))) ** 2} \n')
        out.write('----' * 25)
        out.write('\n')
        out.write('   x      a[i]              f3(Erro)\n')

        for i in range(0, len(point)):
            out.write(f' {point[i]:.3f}    {a[i]}        {abs(Y3[i] - (sol_exact.subs(x,point[i]))) ** 2} \n')
        out.write('----' * 25)
        out.close()

        XX = np.linspace(self.x0, self.xf, 20)
        Ye =  [sol_exact.subs(x,coord) for coord in XX]
        Y1  = [u1.subs(x,coord) for coord in XX]
        Y2  = [u2.subs(x,coord) for coord in XX]
        Y3  = [u3.subs(x,coord) for coord in XX]
        # Plot lists and show them
        plt.plot(XX,Ye, '*', label = 'Exact')
        plt.plot(XX,Y1,'--', label = self.f1)
        plt.plot(XX,Y2,'--', label = self.f2)
        plt.plot(XX,Y3,'--', label = self.f3)

        plt.legend()
        plt.grid()
        title = 'Método da colocação( n = '+str(self.npoint)+')'
        plt.title(title)
        plt.savefig("Imagens/"+title+".png")
        plt.show()


        dsol_exact = sym.diff(sol_exact,x)

        du1 = sym.diff(u1,x)
        du2 = sym.diff(u2,x)
        du3 = sym.diff(u3,x)

        dYe = [dsol_exact.subs(x,coord) for coord in XX]
        dY1 = [du1.subs(x, coord) for coord in XX]
        dY2 = [du2.subs(x, coord) for coord in XX]
        dY3 = [du3.subs(x, coord) for coord in XX]

        # Plot lists and show them
        plt.plot(XX, dYe, '*', label = 'd(Exact)')
        plt.plot(XX, dY1, '--', label = 'd('+self.f1+')')
        plt.plot(XX, dY2, '--', label = 'd('+self.f2+')')
        plt.plot(XX, dY3, '--', label = 'd('+self.f3+')')

        
        plt.legend()
        plt.grid()
        title = 'Método da colocação(Derivadas)( n = '+str(self.npoint)+')'
        plt.title(title)
        plt.savefig("Imagens/"+ title + ".png")
        plt.show()

    def subdomain_method(self):

        # pontos igualmente espaçados #
        point = np.array([i for i in np.linspace(self.x0, self.xf, self.npoint)])
        elem = np.array([[point[i], point[i + 1]] for i in range(self.npoint-1)])

        # f1 = lambda x,n: x ** n * (x - 1.0)
        # u1 = [(lambda x, n=i: f1(x, n)) for i in range(1, n + 1)]

        from sympy.abc import x, y, n


        a = sym.IndexedBase('a')
        u1 = sym.summation(a[n] * eval(self.f1), (n, 1, len(point)-1))
        u2 = sym.summation(a[n] * eval(self.f2), (n, 1, len(point)-1))
        u3 = sym.summation(a[n] * eval(self.f3), (n, 1, len(point)-1))

        r1 = sym.diff(u1, x, x) + u1 - 2 * x
        r2 = sym.diff(u2, x, x) + u2 - 2 * x
        r3 = sym.diff(u3, x, x) + u3 - 2 * x

        R1 = [sym.integrate(r1, (x, el[0], el[1])) for el in elem]
        R2 = [sym.integrate(r2, (x, el[0], el[1])) for el in elem]
        R3 = [sym.integrate(r3, (x, el[0], el[1])) for el in elem]

        out = open("subdomain_method_n="+str(self.npoint)+".txt", "w")  # open output file
        out.write(f'Intervalo de {self.x0} a {self.xf} dividido em {self.npoint} pontos.\n\n')
        out.write("   x      a[n]        f1            f2           f3              Exata\n")

        sol1 = sym.solve(R1, dict=True)
        # print(sol1)
        sol2 = sym.solve(R2, dict=True)
        # print(sol2)
        sol3 = sym.solve(R3, dict=True)
        # print(sol3)

        sol_exact = eval(f_analytic)

        for var in sol1[0]: u1 = u1.subs(var, sol1[0][var])
        for var in sol2[0]: u2 = u2.subs(var, sol2[0][var])
        for var in sol3[0]: u3 = u3.subs(var, sol3[0][var])

        Ye = [sol_exact.subs(x, coord) for coord in point]
        Y1 = [u1.subs(x, coord) for coord in point]
        Y2 = [u2.subs(x, coord) for coord in point]
        Y3 = [u3.subs(x, coord) for coord in point]
        

        for n in range(0, len(point)):
            out.write(f' {point[n - 1]:.3f}    {a[n]}    {str(Y1[n]).rjust(5)[:10]}    {str(Y2[n]).rjust(1)[:10]}    {str(Y3[n]).rjust(1)[:10]}   {sol_exact.subs(x,point[n])} \n')
        out.write('\n \n')
        out.write('   x      a[n]              f1(Erro)\n')

        for i in range(0, len(point)):
            out.write(f' {point[i]:.3f}    {a[i]}        {abs(Y1[i] - (sol_exact.subs(x,point[i]))) ** 2} \n')
        out.write('----'*25)
        out.write('\n')
        out.write('   x      a[i]              f2(Erro)\n')

        for i in range(0, len(point)):
            out.write(f' {point[i]:.3f}    {a[i]}        {abs(Y2[i] - (sol_exact.subs(x,point[i]))) ** 2} \n')
        out.write('----' * 25)
        out.write('\n')
        out.write('   x      a[i]              f3(Erro)\n')

        for i in range(0, len(point)):
            out.write(f' {point[i - 1]:.3f}    {a[i]}        {abs(Y3[i] - (sol_exact.subs(x,point[i]))) ** 2} \n')
        out.write('----' * 25)
        out.close()




        XX = np.linspace(self.x0,self.xf,20)
        Ye = [sol_exact.subs(x, coord) for coord in XX]
        Y1 = [u1.subs(x, coord) for coord in XX]
        Y2 = [u2.subs(x, coord) for coord in XX]
        Y3 = [u3.subs(x, coord) for coord in XX]


        # Plot lists and show them
        plt.plot(XX,Ye, '*', label = 'Exact')
        plt.plot(XX,Y1,'--', label = self.f1)
        plt.plot(XX,Y2,'--', label = self.f2)
        plt.plot(XX,Y3,'--', label = self.f3)

        plt.legend()
        plt.grid()
        title = 'Método dos subdomínios( n = '+str(self.npoint)+')'
        plt.title(title)
        plt.savefig("Imagens/"+title+".png")
        plt.show()


        dsol_exact = sym.diff(sol_exact,x)

        du1 = sym.diff(u1,x)
        du2 = sym.diff(u2,x)
        du3 = sym.diff(u3,x)

        dYe = [dsol_exact.subs(x,coord) for coord in XX]
        dY1 = [du1.subs(x, coord) for coord in XX]
        dY2 = [du2.subs(x, coord) for coord in XX]
        dY3 = [du3.subs(x, coord) for coord in XX]

        # Plot lists and show them
        plt.plot(XX, dYe, '*', label = 'd(Exact)')
        plt.plot(XX, dY1, '--', label = 'd('+self.f1+')')
        plt.plot(XX, dY2, '--', label = 'd('+self.f2+')')
        plt.plot(XX, dY3, '--', label = 'd('+self.f3+')')

        
        plt.legend()
        plt.grid()
        title = 'Método dos subdomínios(Derivadas)( n = '+str(self.npoint)+')'
        plt.title(title)
        plt.savefig("Imagens/"+ title + ".png")
        plt.show()

    def MMQ(self):

        # pontos igualmente espaçados #
        dx = (self.xf - self.x0)/(self.npoint + 1)
        point = np.array([i for i in np.linspace(self.x0+dx, self.xf-dx, self.npoint ) ])
        number = [i for i in range(1,len(point)+1)]
        # f1 = lambda x,n: x ** n * (x - 1.0)
        # u1 = [(lambda x, n=i: f1(x, n)) for i in range(1, n + 1)]

        from sympy.abc import x, y, n


        a = sym.IndexedBase('a')
        u1 = sym.summation(a[n] * eval(self.f1), (n, 1, len(point)))
        u2 = sym.summation(a[n] * eval(self.f2), (n, 1, len(point)))
        u3 = sym.summation(a[n] * eval(self.f3), (n, 1, len(point)))


        r1 = sym.diff(u1, x, x) + u1 - 2*x
        r2 = sym.diff(u2, x, x) + u2 - 2*x
        r3 = sym.diff(u3, x, x) + u3 - 2*x

        R1 = [sym.integrate(r1*(sym.diff(r1,a[n])),(x,self.x0,self.xf)) for n in number]
        R2 = [sym.integrate(r2*(sym.diff(r2,a[n])),(x,self.x0,self.xf)) for n in number]
        R3 = [sym.integrate(r3*(sym.diff(r3,a[n])),(x,self.x0,self.xf)) for n in number]

        out = open("MMQ_comparation_n="+str(self.npoint)+".txt", "w")  # open output file
        out.write(f'Intervalo de {self.x0} a {self.xf} dividido em {self.npoint} pontos.\n\n')
        out.write("   x      a[n]        f1            f2           f3              Exata\n")

        sol1 = sym.solve(R1, dict=True)
        # print(sol1)
        sol2 = sym.solve(R2, dict=True)
        # print(sol2)
        sol3 = sym.solve(R3, dict=True)
        # print(sol3)

        sol_exact = eval(f_analytic)

        
        
        for var in sol1[0]: u1 = u1.subs(var, sol1[0][var])
        for var in sol2[0]: u2 = u2.subs(var, sol2[0][var])
        for var in sol3[0]: u3 = u3.subs(var, sol3[0][var])

        Ye =  [sol_exact.subs(x,coord) for coord in point]
        Y1  = [u1.subs(x,coord) for coord in point]
        Y2  = [u2.subs(x,coord) for coord in point]
        Y3  = [u3.subs(x,coord) for coord in point]

        for n in range(0, len(point)):
            out.write(f' {point[n - 1]:.3f}    {a[n]}    {str(Y1[n]).rjust(5)[:10]}    {str(Y2[n]).rjust(1)[:10]}    {str(Y3[n]).rjust(1)[:10]}   {sol_exact.subs(x,point[n])} \n')
        out.write('\n \n')
        out.write('   x      a[n]              f1(Erro)\n')

        for i in range(0, len(point)):
            out.write(f' {point[i]:.3f}    {a[i]}        {abs(Y1[i] - (sol_exact.subs(x,point[i]))) ** 2} \n')
        out.write('----'*25)
        out.write('\n')
        out.write('   x      a[i]              f2(Erro)\n')

        for i in range(0, len(point)):
            out.write(f' {point[i]:.3f}    {a[i]}        {abs(Y2[i] - (sol_exact.subs(x,point[i]))) ** 2} \n')
        out.write('----' * 25)
        out.write('\n')
        out.write('   x      a[i]              f3(Erro)\n')

        for i in range(0, len(point)):
            out.write(f' {point[i - 1]:.3f}    {a[i]}        {abs(Y3[i] - (sol_exact.subs(x,point[i]))) ** 2} \n')
        out.write('----' * 25)
        out.close()



        XX = np.linspace(self.x0, self.xf, 20)
        Ye =  [sol_exact.subs(x,coord) for coord in XX]
        Y1  = [u1.subs(x,coord) for coord in XX]
        Y2  = [u2.subs(x,coord) for coord in XX]
        Y3  = [u3.subs(x,coord) for coord in XX]


        # Plot lists and show them
        plt.plot(XX,Ye, '*', label = 'Exact')
        plt.plot(XX,Y1,'--', label = self.f1)
        plt.plot(XX,Y2,'--', label = self.f2)
        plt.plot(XX,Y3,'--', label = self.f3)

        plt.legend()
        plt.grid()
        title = 'MMQ( n = '+str(self.npoint)+')'
        plt.title(title)
        plt.savefig("Imagens/"+title+".png")
        plt.show()


        dsol_exact = sym.diff(sol_exact,x)

        du1 = sym.diff(u1,x)
        du2 = sym.diff(u2,x)
        du3 = sym.diff(u3,x)

        dYe = [dsol_exact.subs(x,coord) for coord in XX]
        dY1 = [du1.subs(x, coord) for coord in XX]
        dY2 = [du2.subs(x, coord) for coord in XX]
        dY3 = [du3.subs(x, coord) for coord in XX]

        # Plot lists and show them
        plt.plot(XX, dYe, '*', label = 'd(Exact)')
        plt.plot(XX, dY1, '--', label = 'd('+self.f1+')')
        plt.plot(XX, dY2, '--', label = 'd('+self.f2+')')
        plt.plot(XX, dY3, '--', label = 'd('+self.f3+')')

        
        plt.legend()
        plt.grid()
        title = 'MMQ(Derivadas)( n = '+str(self.npoint)+')'
        plt.title(title)
        plt.savefig("Imagens/"+ title + ".png")
        plt.show()




if __name__ == '__main__':

    x0 = 0.0
    xf = 1.0
    n = [3]

    
    # Funções escolhidas
    f1 = 'x*(x - self.xf) ** n'
    f2 = '(x ** n) * (x - 1.0)'
    f3 = '(1 - x) ** n * (1.0 - sym.exp(x))'  
    f = [ f1, f2, f3 ]
    # função analitica
    f_analytic = '2*x - (2*sym.sin(x))/np.sin(1)'
    # criando objeto
    # problema = Problem(x0, xf, n1, f, f_analytic)

    problemas = [Problem(x0, xf, i, f, f_analytic) for i in n]
    

    # Chamando o método
    for i in range(len(n)):
        problemas[i].collocation_method()
        # problemas[i].subdomain_method()
        # problemas[i].MMQ()



