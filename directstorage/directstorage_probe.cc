// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <Windows.h>
#include <dstorage.h>
#include <wrl/client.h>
#include <iostream>

int main() {
  Microsoft::WRL::ComPtr<IDStorageFactory> factory;
  const HRESULT result = DStorageGetFactory(IID_PPV_ARGS(&factory));
  std::cout << "DStorageGetFactory HRESULT: 0x" << std::hex
            << static_cast<unsigned long>(result) << std::endl;
  for (const auto* name : {L"dstorage.dll", L"dstoragecore.dll"}) {
    const auto module = GetModuleHandleW(name);
    wchar_t path[32768]{};
    if (module) GetModuleFileNameW(module, path, 32768);
    std::wcout << name << L": " << path << std::endl;
  }
  return FAILED(result) ? 1 : 0;
}
