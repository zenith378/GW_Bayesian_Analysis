using Pkg
using StatsBase
using CSV
using DelimitedFiles
using Plots

# Gravitational wave strain for GW150914_R1 for H1 (see http://losc.ligo.org)
# This file has 4096 samples per second
# starting GPS 1126257415 duration 4096
df = CSV.read("H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt",  DataFrame, header=[:logt90])
myarray=readdlm("H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt",Float64)

function build_matrix(x::Matrix{T}) where T
    n = size(x, 1)
    C = zeros(T, n, n)

    for i in 1:n
        for j in 1:n
            for k in 0:(n-1)
                # Gestione delle condizioni di boundary
                idx_i = mod1(i + k, n)
                idx_j = mod1(j + k, n)

                C[i, j] += x[idx_i, 1] * x[idx_j, 1]
            end
            C[i, j] /= n
        end
    end

    return C
end

result_matrix = build_matrix(myarray)

println(result_matrix)

function calculate_autocorrelation(a)
    N = length(a)
    rho_i_array = Float64[]

    for i in 0:N-1
        rho_i = 0.0
        for j in 1:N
            rho_i += a[j] * a[(j+i-1) % N + 1]
        end
        push!(rho_i_array, rho_i)
    end

    return rho_i_array
end

rho_results = calculate_autocorrelation(myarray)

println("Risultati di rho_i per ogni i:", rho_results)

a = copy(myarray)
matrice = reshape(a, 4096, 4096)
plot(range(1,4096,4096),matrice[:, 1])
plot!(range(1,4096,4096),matrice[:, 2])
plot!(range(1,4096,4096),matrice[:, 3])
x = range(0, 69, length=16777216)
autocorr = autocor(matrice,[size(matrice,1)-1])
t = range(0,72,73)
mean_per_row = mean(matrice, dims=1)
plot(range(1,4096,4096),autocorr[1,:])