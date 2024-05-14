@inline function apply_act(::typeof(identity), x::Any)
    x
end

@inline function apply_act(activation::Any, x::Number)
    activation(x)
end

@inline function apply_act(activation::Any, x::AbstractArray)
    activation.(x)
end
