using PyPlot

L = 1001
delta = .1
tau = 0.001
m = 10000
D = 1.

# Prepare arrays
phi = zeros(Float64, L)
variance = zeros(Float64, m)



i0 = (L+1)/2
weights = [1:L] - i0
sqweights = weights .* weights

function var(x::Array{Float64})
        xsum = sum(x)
        variance = sum(x.*sqweights) / xsum - (sum(x.*weights) / xsum)^2
        return variance
end

# Boundary conditions:
phi[(L+1)/2] = 1
# phi[1] = 1

function diffuse()
    # Prepare some values for the evolution
    expa = exp(-tau*D/delta/delta)
    expa2 = exp(-tau*D/delta/delta*0.5)
    expmat = 0.5 * [[1+expa 1-expa], [1-expa 1+expa]]
    expmat2 = 0.5 * [[1+expa2 1-expa2], [1-expa2 1+expa2]]
    even = iseven(L)

    variance[1] = var(phi)

    # Perform time evolution
    for i=1:m-1
        # First A step
        for j=1:2:L-1
            phi[j:j+1] = expmat2*phi[j:j+1]
        end
        if ~even
            phi[L] = phi[L] * expa2
        end
        # B step
        phi[1] = phi[1] * expa
        for j=2:2:L-1
            phi[j:j+1] = expmat*phi[j:j+1]
        end
        if even
            phi[L] = phi[L] * expa
        else
            # second A step
            phi[L] = phi[L] * expa2
        end
        for j=1:2:L-1
            phi[j:j+1] = expmat*phi[j:j+1]
        end

        # Now sample
        variance[i+1] = var(phi)
    end
    return variance
end
@elapsed variance = diffuse()
t = [0:m-1] * tau

plot(t, variance)
title("Variance")
xlabel("Time")
ylabel("Variance")
