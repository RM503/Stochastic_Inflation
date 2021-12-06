"""
This code implements a stochastic RK2 method for solving the dynamics of stochastic
inflation using the Deformed Starobinsky potential with slow-roll noise terms.

Potential: V(φ) = V0*(1 + ξ - exp(-αφ) - ξ exp(-βφ²))² 
"""
using Random
using StatsBase
using PyPlot
pygui(true)

α = sqrt(2.0/3)
β = 1.114905
ξ = -0.480964
V0 = 1.27*10^-9
dN = 0.001

function V(x)
    return @. V0*(1+ξ-exp(-α*x)-ξ*exp(-β*x^2))^2
end

function dV(x)
    return @. 2*V0*(1+ξ-exp(-α*x)-ξ*exp(-β*x^2))*(α*exp(-α*x)+2*β*ξ*x*exp(-β*x^2))
end

function back_evolve(ϕ_in, ℯfolds)
    N_test = ℯfolds
    n = Int(round(N_test/dN))
    ϕ = zeros(n)
    dϕ = zeros(n)
    ϕ[1] = ϕ_in
    dϕ[1] = -dV(ϕ_in)/V(ϕ_in)

    """
    RK2 code for solving the inflaton background evolution
    """
    for i in 1:1:n-1
        K1 = dN*dϕ[i]
        L1 = -dN*(3*dϕ[i]-0.5*dϕ[i]^3)-dN*(3-0.5*dϕ[i]^2)*dV(ϕ[i])/V(ϕ[i])

        K2 = dN*(dϕ[i]+L1)
        L2 = -dN*(3*(dϕ[i]+L1)-0.5*(dϕ[i]+L1)^3)-dN*(3-0.5*(dϕ[i]+L1)^2)*dV(ϕ[i]+K1)/V(ϕ[i]+K1)

        ϕ[i+1] = ϕ[i] + 0.5*(K1 + K2)
        dϕ[i+1] = dϕ[i] + 0.5*(L1 + L2)
    end
    return ϕ, dϕ
end

N = LinRange(0, 70, Int(round(70/dN)))
ϕ_in = 5.82

ϕ, dϕ = back_evolve(ϕ_in, 70)
ϵ1 = @. 0.5*dϕ^2
H = @. sqrt(V(ϕ)/(3-ϵ1))

ϕbar = zeros(length(N))
dϕbar = zeros(length(N))
ϕbar[1] = ϕ_in
dϕbar[1] = -dV(ϕ_in)/V(ϕ_in)

Φ = zeros(length(N))
DΦ = zeros(length(N))
Φ[1] = ϕ_in
DΦ[1] = -dV(ϕ_in)/V(ϕ_in)

F = randn(length(N))/sqrt(dN)
S = sample([-1,1], Weights([0.5,0.5]), length(N))
#Compare solutions using Euler-Maruyama and Stochastic RK2
for j in 1:1:length(N)-1

    """
    Euler-Maruyama discretization
    """
    ξ_ϕ = (H[j]/(2*π))*F[j]
    ϕbar[j+1] = ϕbar[j] + dϕbar[j]*dN + dN*ξ_ϕ
    dϕbar[j+1] = ( dϕbar[j] - dN*3*dϕbar[j] + dN*0.5*dϕbar[j]^3 - dN*(3-0.5*dϕbar[j]^2)
                *( dV(ϕbar[j])/V(ϕbar[j]) ) )
    """
    A stochastic RK algorithm that reduces to RK2 in a straightforward manner
    Ref.  arXiv:1210.0933 [math.NA]
    """
    𝓀1 = dN*DΦ[j] + (dN*F[j] - S[j]*sqrt(dN))*(H[j]/(2*π))
    𝓁1 = -dN*(3*DΦ[j]-0.5*DΦ[j]^3)-dN*(3-0.5*DΦ[j]^2)*dV(Φ[j])/V(Φ[j])
    𝓀2 = dN*(DΦ[j] + 𝓁1) + (dN*F[j] + S[j]*sqrt(dN))*(H[j+1]/(2*π))
    𝓁2 = -dN*(3*(DΦ[j]+𝓁1)-0.5*(DΦ[j]+𝓁1)^3)-dN*(3-0.5*(DΦ[j]+𝓁1)^2)*dV(Φ[j]+𝓀1)/V(Φ[j]+𝓀1)

    Φ[j+1] = Φ[j] + 0.5*(𝓀1 + 𝓀2)
    DΦ[j+1] = DΦ[j] + 0.5*(𝓁1 + 𝓁2)
end
