@setup_workload begin
    @compile_workload begin
        rng = Random.default_rng()
        mts = Type{<:AbstractICNF}[RNODE, FFJORD, Planar]
        cmts = Type{<:AbstractCondICNF}[CondRNODE, CondFFJORD, CondPlanar]
        cmodes = Type{<:ComputeMode}[ADVectorMode, ZygoteMatrixMode]
        nvars = 2
        r = rand(Float32, nvars)
        r_arr = rand(Float32, nvars, 2)
        r2 = rand(Float32, nvars)
        r2_arr = rand(Float32, nvars, 2)

        for cmode in cmodes, mt in mts
            if mt <: Planar
                nn = PlanarLayer(nvars, tanh)
            else
                nn = Lux.Dense(nvars => nvars, tanh)
            end
            icnf = construct(mt, nn, nvars; compute_mode = cmode)
            ps, st = Lux.setup(rng, icnf)
            ps = ComponentArray(ps)
            if cmode <: VectorMode
                L = loss(icnf, r, ps, st)
            elseif cmode <: MatrixMode
                L = loss(icnf, r_arr, ps, st)
            end
        end
        for cmode in cmodes, mt in cmts
            if mt <: CondPlanar
                nn = PlanarLayer(nvars, tanh; cond = true)
            else
                nn = Lux.Dense(2 * nvars => nvars, tanh)
            end
            icnf = construct(mt, nn, nvars; compute_mode = cmode)
            ps, st = Lux.setup(rng, icnf)
            ps = ComponentArray(ps)
            if cmode <: VectorMode
                L = loss(icnf, r, r2, ps, st)
            elseif cmode <: MatrixMode
                L = loss(icnf, r_arr, r2_arr, ps, st)
            end
        end
    end
end
