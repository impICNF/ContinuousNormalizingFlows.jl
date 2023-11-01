@setup_workload begin
    @compile_workload begin
        fllprcmpltn = @load_preference("fullprecompilation", false)
        mts = ifelse(
            fllprcmpltn,
            Type{<:AbstractFlows}[RNODE, FFJORD, Planar, CondRNODE, CondFFJORD, CondPlanar],
            Type{<:AbstractFlows}[RNODE],
        )
        compute_modes = ifelse(
            fllprcmpltn,
            Type{<:ComputeMode}[
                ADVecJacVectorMode,
                ADJacVecVectorMode,
                ZygoteVectorMode,
                SDVecJacMatrixMode,
                SDJacVecMatrixMode,
                ZygoteMatrixMode,
            ],
            Type{<:ComputeMode}[ZygoteMatrixMode],
        )
        omodes = Mode[TrainMode(), TestMode()]
        nvars_ = Int[1]
        inplaces = Bool[false]
        aug_steers = Bool[true]
        nvars_ = ifelse(fllprcmpltn, Int[1, 2], Int[1])
        inplaces = ifelse(fllprcmpltn, Bool[false, true], Bool[false])
        aug_steers = ifelse(fllprcmpltn, Bool[false, true], Bool[true])
        data_type = Float32

        for compute_mode in compute_modes,
            inplace in inplaces,
            aug_steer in aug_steers,
            nvars in nvars_,
            omode in omodes,
            mt in mts

            data_dist = Beta{data_type}(convert(Tuple{data_type, data_type}, (2, 4))...)
            r = convert.(data_type, rand(data_dist, nvars))
            if compute_mode <: MatrixMode
                r = hcat(r)
            end
            data_dist2 = Beta{data_type}(convert(Tuple{data_type, data_type}, (4, 2))...)
            r2 = convert.(data_type, rand(data_dist2, nvars))
            if compute_mode <: MatrixMode
                r2 = hcat(r2)
            end

            nn = ifelse(
                mt <: AbstractCondICNF,
                ifelse(
                    mt <: CondPlanar,
                    ifelse(
                        aug_steer,
                        PlanarLayer(nvars * 2, tanh; n_cond = nvars),
                        PlanarLayer(nvars, tanh; n_cond = nvars),
                    ),
                    ifelse(
                        aug_steer,
                        Lux.Dense(nvars * 3 => nvars * 2, tanh),
                        Lux.Dense(nvars * 2 => nvars, tanh),
                    ),
                ),
                ifelse(
                    mt <: Planar,
                    ifelse(
                        aug_steer,
                        PlanarLayer(nvars * 2, tanh),
                        PlanarLayer(nvars, tanh),
                    ),
                    ifelse(
                        aug_steer,
                        Lux.Dense(nvars * 2 => nvars * 2, tanh),
                        Lux.Dense(nvars => nvars, tanh),
                    ),
                ),
            )
            icnf = ifelse(
                aug_steer,
                construct(
                    mt,
                    nn,
                    nvars,
                    nvars;
                    data_type,
                    compute_mode,
                    inplace,
                    steer_rate = convert(data_type, 0.1),
                ),
                construct(mt, nn, nvars; data_type, compute_mode, inplace),
            )
            ps, st = Lux.setup(icnf.rng, icnf)
            ps = ComponentArray(ps)
            L =
                mt <: AbstractCondICNF ? loss(icnf, omode, r, r2, ps, st) :
                loss(icnf, omode, r, ps, st)
        end
    end
end
