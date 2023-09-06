@setup_workload begin
    @compile_workload begin
        fllprcmpltn = @load_preference("fullprecompilation", false)
        mts =
            fllprcmpltn ? Type{<:AbstractICNF}[RNODE, FFJORD, Planar] :
            Type{<:AbstractICNF}[RNODE]
        cmts =
            fllprcmpltn ? Type{<:AbstractCondICNF}[CondRNODE, CondFFJORD, CondPlanar] :
            Type{<:AbstractCondICNF}[]
        cmodes =
            fllprcmpltn ?
            Type{<:ComputeMode}[ADVectorMode, ZygoteMatrixMode, SDVecJacMatrixMode] :
            Type{<:ComputeMode}[ZygoteMatrixMode]
        omodes = Mode[TrainMode(), TestMode()]
        nvars = 1
        r = rand(Float32, nvars)
        r_arr = hcat(r)
        r2 = rand(Float32, nvars)
        r2_arr = hcat(r)

        for cmode in cmodes, omode in omodes, mt in mts
            if mt <: Planar
                nn = PlanarLayer(nvars, tanh)
            else
                nn = Lux.Dense(nvars => nvars, tanh)
            end
            icnf = construct(mt, nn, nvars; compute_mode = cmode)
            ps, st = Lux.setup(icnf.rng, icnf)
            ps = ComponentArray(ps)
            if cmode <: VectorMode
                L = loss(icnf, omode, r, ps, st)
            elseif cmode <: MatrixMode
                L = loss(icnf, omode, r_arr, ps, st)
            end
        end
        for cmode in cmodes, omode in omodes, mt in cmts
            if mt <: CondPlanar
                nn = PlanarLayer(nvars, tanh; n_cond = nvars)
            else
                nn = Lux.Dense(2 * nvars => nvars, tanh)
            end
            icnf = construct(mt, nn, nvars; compute_mode = cmode)
            ps, st = Lux.setup(icnf.rng, icnf)
            ps = ComponentArray(ps)
            if cmode <: VectorMode
                L = loss(icnf, omode, r, r2, ps, st)
            elseif cmode <: MatrixMode
                L = loss(icnf, omode, r_arr, r2_arr, ps, st)
            end
        end
    end
end
