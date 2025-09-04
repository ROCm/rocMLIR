//===-- JSONTransport.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/JSONTransport.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

char TransportUnhandledContentsError::ID;

TransportUnhandledContentsError::TransportUnhandledContentsError(
    std::string unhandled_contents)
    : m_unhandled_contents(unhandled_contents) {}

<<<<<<< HEAD
  if (timeout && timeout_supported) {
    SelectHelper sh;
    sh.SetTimeout(*timeout);
    sh.FDSetRead(
        reinterpret_cast<lldb::socket_t>(descriptor.GetWaitableHandle()));
    Status status = sh.Select();
    if (status.Fail()) {
      // Convert timeouts into a specific error.
      if (status.GetType() == lldb::eErrorTypePOSIX &&
          status.GetError() == ETIMEDOUT)
        return make_error<TransportTimeoutError>();
      return status.takeError();
    }
  }

  std::string data;
  data.resize(length);
  Status status = descriptor.Read(data.data(), length);
  if (status.Fail())
    return status.takeError();

  // Read returns '' on EOF.
  if (length == 0)
    return make_error<TransportEOFError>();

  // Return the actual number of bytes read.
  return data.substr(0, length);
=======
void TransportUnhandledContentsError::log(llvm::raw_ostream &OS) const {
  OS << "transport EOF with unhandled contents: '" << m_unhandled_contents
     << "'";
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
}
std::error_code TransportUnhandledContentsError::convertToErrorCode() const {
  return std::make_error_code(std::errc::bad_message);
}
