/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Exercise the graph API used during EP import, which hip-mlir-opt cannot
// reach. A constant result does not imply a runtime input; existing inputs
// must still be returned in order.
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include "mlir-graph.hpp"
#include "mlir-model.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "morphizen/morphizen-ort-api-ext.hpp"
#include "llvm/Support/raw_ostream.h"

namespace morphizen {
extern "C" MorphizenOrtApiExt *get_morphizen_ort_api_mlir();
}

int main() {
  Ort::InitApi();
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  const char *graphs[] = {
      R"mlir(module {
        func.func @main_graph() -> tensor<1xf32> {
          %c = "onnx.Constant"() {value = dense<2.0> : tensor<1xf32>} : () -> tensor<1xf32>
          func.return %c : tensor<1xf32>
        }
      })mlir",
      R"mlir(module {
        func.func @main_graph(%x: tensor<1xf32>) -> tensor<1xf32> {
          func.return %x : tensor<1xf32>
        }
      })mlir",
      R"mlir(module {
        func.func @main_graph(%x: tensor<1xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
          func.return %y : tensor<2xf32>
        }
      })mlir"};
  auto *api = morphizen::get_morphizen_ort_api_mlir();
  for (size_t count = 0; count != 3; ++count) {
    auto module =
        mlir::parseSourceString<mlir::ModuleOp>(graphs[count], &context);
    if (!module)
      return 1;
    auto model = morphizen::mlir_impl::MLIRModel::create(std::move(module));
    auto &graph = model->main_graph();
    auto &opaque = reinterpret_cast<const morphizen::Graph &>(graph);
    auto inputs = api->graph_get_inputs_unsafe(opaque);
    auto outputs = api->graph_get_outputs_unsafe(opaque);
    if (!inputs.get() || inputs->size() != count || outputs->size() != 1) {
      llvm::errs() << "Unexpected graph API input/output count\n";
      return 1;
    }
    auto expected = graph.get_inputs();
    for (size_t i = 0; i != count; ++i)
      if ((*inputs)[i] != reinterpret_cast<const morphizen::NodeArg *>(
                              expected[i].to_morphizen_core_node_arg_ptr()))
        return 1;
  }
  llvm::outs() << "PASS: zero, one and two runtime inputs; outputs preserved\n";
  return 0;
}
