module diffusion
    type diffusor
        phi::Array{Float}
        L::Int
        D::Float
        delta::Float
        tau::Float
        m::Int

        expa::Float
        expb::Float

        function diffusor(L::int, D, delta, tau, m)
            phi = zeros(Float,L)
            new(phi,L,D,delta,tau,m)
        end
    end

    function evolve(di::diffusor)

    end
end
