/* Copyright (C) 2026 zk4x
 * SPDX-License-Identifier: Apache-2.0
 *
 * The smallest useful kernel: one 32x32 tile from DRAM, through L1 and the
 * cores, and back to DRAM.
 *
 *   export LD_LIBRARY_PATH="$PWD/target/release:$LD_LIBRARY_PATH"
 *   export CPATH="$PWD/target/release:$CPATH"
 *   gcc dram_loopback.c -o dram_loopback && ./dram_loopback
 *
 * The device comes from the environment: `TT_HARDWARE=1` for the card, or
 * `$TT_METAL_SIMULATOR` for the simulator. With neither, `ttInit` refuses.
 */

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "skamandros.h"

/* DRAM -> L1 -> DRAM, one tile: `%in` and `%out` are 1024 f32 each. */
static const char PROGRAM[] =
    "version 9\n"
    "grid 1,1\n"
    "%in = param f32 1024\n"
    "%out = param f32 1024\n"
    "%b0 = buffer f32 32,32\n"
    "copy 32,32 %b0 [0] = %in [0]\n"
    "copy 32,32 %out [0] = %b0 [0]\n";

/* The ABI's entry points, as `dlopen` hands them over. */
typedef ttStatus (*init_fn)(char **err);
typedef ttStatus (*release_fn)(char **err);
typedef ttStatus (*compile_fn)(const char *text, size_t len, ttModule **out, char **err);
typedef ttStatus (*unload_fn)(ttModule *module, char **err);
typedef ttStatus (*alloc_fn)(size_t bytes, uint64_t *addr, char **err);
typedef ttStatus (*h2d_fn)(uint64_t dst, const uint8_t *src, size_t len, char **err);
typedef ttStatus (*launch_fn)(ttModule *module, const uint64_t *params, size_t nparams, char **err);
typedef void (*free_fn)(void *ptr);

static void *symbol(void *lib, const char *name) {
    void *at = dlsym(lib, name);
    if (at == NULL) {
        fprintf(stderr, "dlsym(%s): %s\n", name, dlerror());
        exit(EXIT_FAILURE);
    }
    return at;
}

static void check(ttStatus status, char *err, const char *what, free_fn tt_free) {
    if (status == TT_OK) {
        return;
    }
    fprintf(stderr, "%s failed: %s\n", what, err != NULL ? err : "(no message)");
    if (err != NULL) {
        tt_free(err);
    }
    exit(EXIT_FAILURE);
}

int main(void) {
    void *lib = dlopen("libskamandros.so", RTLD_NOW);
    if (lib == NULL) {
        fprintf(stderr, "dlopen(libskamandros.so): %s\n", dlerror());
        fprintf(stderr, "export LD_LIBRARY_PATH to the directory holding it\n");
        return EXIT_FAILURE;
    }

    free_fn tt_free;
    init_fn init;
    release_fn release;
    compile_fn compile;
    unload_fn unload;
    alloc_fn alloc;
    h2d_fn h2d;
    launch_fn launch;
    *(void **)&tt_free = symbol(lib, "ttFree");
    *(void **)&init = symbol(lib, "ttInit");
    *(void **)&release = symbol(lib, "ttRelease");
    *(void **)&compile = symbol(lib, "ttCompile");
    *(void **)&unload = symbol(lib, "ttModuleUnload");
    *(void **)&alloc = symbol(lib, "ttMemAlloc");
    *(void **)&h2d = symbol(lib, "ttMemcpyH2D");
    *(void **)&launch = symbol(lib, "ttLaunch");

    char *err = NULL;
    check(init(&err), err, "ttInit", tt_free);

    ttModule *module = NULL;
    check(compile(PROGRAM, sizeof(PROGRAM) - 1, &module, &err), err, "ttCompile", tt_free);

    uint64_t input = 0;
    uint64_t output = 0;
    check(alloc(sizeof(float) * 1024, &input, &err), err, "ttMemAlloc(input)", tt_free);
    check(alloc(sizeof(float) * 1024, &output, &err), err, "ttMemAlloc(output)", tt_free);

    float data[1024];
    for (int i = 0; i < 1024; i++) {
        data[i] = (float)i * 0.25f;
    }
    check(h2d(input, (const uint8_t *)data, sizeof(data), &err), err, "ttMemcpyH2D", tt_free);

    const uint64_t params[] = {input, output};
    check(launch(module, params, 2, &err), err, "ttLaunch", tt_free);

    check(unload(module, &err), err, "ttModuleUnload", tt_free);
    check(release(&err), err, "ttRelease", tt_free);
    dlclose(lib);
    puts("dram loopback: one 32x32 tile from DRAM through L1 and back");
    return EXIT_SUCCESS;
}
