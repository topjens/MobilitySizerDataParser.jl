module MobilitySizerDataParser

using DataFrames
using Dates
using Plots
using DifferentialMobilityAnalyzers
using CSV

export parse_file

function parse_file(path::AbstractString)
    outdir = dirname(path)
    lines = readlines(path)

    i = 1
    block_id = 0

    while i <= length(lines)
        tsline = strip(lines[i])

        m = match(r"'([^']+)'\s+'([^']+)'", tsline)
        m === nothing && error("Invalid timestamp line at line $i")

        start_ts = DateTime(m.captures[1], "mm-dd-yyyy HH:MM:SS")
        stop_ts  = DateTime(m.captures[2], "mm-dd-yyyy HH:MM:SS")

        i += 1
        block_id += 1

        println("Processing $(block_id)")

        vals = split(strip(lines[i]))

        length(vals) == 15 || error("Expected 15 values at line $i")

        qsh = parse(Float64, vals[1]) / 60. / 1000.
        qsa = parse(Float64, vals[2]) / 60. / 1000.
        r₁ = parse(Float64, vals[3])
        r₂ = parse(Float64, vals[4])
        l = parse(Float64, vals[5])
        # 6
        # 7
        p = parse(Float64, vals[8]) * 100.0
        # 9
        t = parse(Float64, vals[10]) + 273.15
        
        
        leff = 0.0
        m = 6
        DMAtype = :cylindrical
        polarity = :+

        Λ = DMAconfig(t, p, qsa, qsh, r₁, r₂, l, leff, polarity, m, DMAtype)

        i += 1

        xy_rows = NamedTuple[]

        while i <= length(lines)

            line = strip(lines[i])

            # next block starts with quote
            startswith(line, "'") && break

            parts = split(line)

            length(parts) >= 2 || error("Invalid x y line at line $i")

            V = parse(Float64, parts[1])
            N = parse(Float64, parts[2])

            push!(xy_rows, (
                V = V,
                Rcn = N,
            ))

            i += 1
        end

        experiment = DataFrame(xy_rows)
        experiment[!,:Dp] = ztod(Λ, 1, vtoz(Λ, experiment[!,:V]))
        δ = setupSMPSdata(Λ, experiment[!,:V])

        𝕣 = (experiment,:Dp,:Rcn,δ) |> interpolateDataFrameOntoδ
        # 𝕟ⁱⁿᵛ = rinv(𝕣.N, δ, λ₁=0.1, λ₂=1.0)
        𝕟ⁱⁿᵛ² = rinv2(𝕣.N, δ, λ₁=0.1, λ₂=1.0)

        inverted = DataFrame(Dp_inv=𝕟ⁱⁿᵛ².Dp, N=𝕟ⁱⁿᵛ².N)

        # p1 = scatter(experiment[!,:Dp], experiment[!,:Rcn],
        #              xscale = :log10,
        #              xlabel = "D_p [nm]",
        #              ylabel = "raw number [-]",
        #              title = "Raw number of particles",
        #              legend = false)

        # p2 = scatter(𝕟ⁱⁿᵛ.Dp, 𝕟ⁱⁿᵛ.N,
        #              xscale = :log10,
        #              xlabel = "D_p [nm]",
        #              ylabel = "dN/dlogD_p [cm-3]",
        #              title = "Inversed number distribution",
        #              legend = false)

        # p3 = scatter(𝕟ⁱⁿᵛ².Dp, 𝕟ⁱⁿᵛ².N,
        #              xscale = :log10,
        #              xlabel = "D_p [nm]",
        #              ylabel = "dN/dlogD_p [cm-3]",
        #              title = "Inversed number distribution 2",
        #              legend = false)

        # # Arrange side-by-side
        # p = plot(p1, p2, p3, layout = (1, 3), size=(1200, 400),  display=false);

        # savefig(p, joinpath(outdir, "plot_$(block_id).png"))
        CSV.write(joinpath(outdir, "raw_$(block_id).csv"), experiment)
        CSV.write(joinpath(outdir, "inverted_$(block_id).csv"), inverted)
    end
end

function process_file(path::AbstractString)
    outdir = dirname(path)
    lines = readlines(path)

    i = 1
    block_id = 0

    ts = DateTime[]
    geomean = Float64[]
    N = Float64[]

    while i <= length(lines)
        tsline = strip(lines[i])

        m = match(r"'([^']+)'\s+'([^']+)'", tsline)
        m === nothing && error("Invalid timestamp line at line $i")

        start_ts = DateTime(m.captures[1], "mm-dd-yyyy HH:MM:SS")
        stop_ts  = DateTime(m.captures[2], "mm-dd-yyyy HH:MM:SS")

        avg_ts = start_ts + (stop_ts - start_ts) / 2

        i += 1
        block_id += 1

        println("Processing $(block_id)")

        vals = split(strip(lines[i]))

        length(vals) == 15 || error("Expected 15 values at line $i")

        qsh = parse(Float64, vals[1]) / 60. / 1000.
        qsa = parse(Float64, vals[2]) / 60. / 1000.
        r₁ = parse(Float64, vals[3])
        r₂ = parse(Float64, vals[4])
        l = parse(Float64, vals[5])
        # 6
        # 7
        p = parse(Float64, vals[8]) * 100.0
        # 9
        t = parse(Float64, vals[10]) + 273.15
        
        
        leff = 13.0
        m = 6
        DMAtype = :cylindrical
        polarity = :+

        Λ = DMAconfig(t, p, qsa, qsh, r₁, r₂, l, leff, polarity, m, DMAtype)

        i += 1

        xy_rows = NamedTuple[]

        while i <= length(lines)

            line = strip(lines[i])

            # next block starts with quote
            startswith(line, "'") && break

            parts = split(line)

            length(parts) >= 2 || error("Invalid x y line at line $i")

            V = parse(Float64, parts[1])
            N = parse(Float64, parts[2])

            push!(xy_rows, (
                V = V,
                Rcn = N,
            ))

            i += 1
        end

        experiment = DataFrame(xy_rows)
        experiment[!,:Dp] = ztod(Λ, 1, vtoz(Λ, experiment[!,:V]))
        δ = setupSMPSdata(Λ, experiment[!,:V])

        w = (experiment,:Dp,:Rcn,δ) |> interpolateDataFrameOntoδ
        x = rinv2(w.N, δ, λ₁=0.1, λ₂=1.0)

        push!(ts, avg_ts)
        push!(geomean, exp(sum(w .* log.(x))/sum(w)))
        push!(N, sum(w.N .* w.ΔlnD))
    end

    processed_data = DataFrame(t=ts, geomean=geomean, N=N)
    CSV.write(joinpath(outdir, "processed_data.csv"), processed_data)
    
end

end
