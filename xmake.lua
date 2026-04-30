set_project("SPH 2D")

add_rules("mode.debug", "mode.release")
add_requires("raylib")

target("benchmark")
    set_kind("binary")

    add_includedirs("include")

    add_files("src/main.cpp")
    add_packages("raylib")
    set_languages("cxx17")

    add_options("-O3")
    

-- spatial hash algorithm
target("v1")
    set_kind("binary")

    add_includedirs("include")

    add_files("src/main_v1.cpp")
    add_packages("raylib")
    set_languages("cxx17")

    if is_plat("windows") then
        add_cxflags("/openmp")
    else
        add_cxflags("-fopenmp")
        add_ldflags("-fopenmp")
    end

    add_options("-O3")
    
-- SoA, reordering, spatial hash
target("v2")
    set_kind("binary")

    add_includedirs("include")

    add_files("src/main_v2.cpp")
    add_packages("raylib")
    set_languages("cxx17")

    if is_plat("windows") then
        add_cxflags("/openmp")
    else
        add_cxflags("-fopenmp")
        add_ldflags("-fopenmp")
    end

    add_options("-O3")
    
    
-- SoA, reordering, spatial hash, poly6 kernel, SIMD
target("v3")
    set_kind("binary")

    add_includedirs("include")

    add_files("src/main_v3.cpp")
    add_packages("raylib")
    set_languages("cxx17")
    set_optimize("fastest")

    if is_plat("windows") then
        -- enable OpenMP on Windows
        add_cxflags("/openmp")
        -- enable AVX512 on Windows
        add_cxflags("/arch:AVX2", "/fp:fast")
    else
        add_cxflags("-fopenmp")
        add_ldflags("-fopenmp")

        -- on gcc, g++
        add_cxflags("-mavx2", "-mfma")
    end

    set_symbols("debug")
    add_vectorexts("avx2")

