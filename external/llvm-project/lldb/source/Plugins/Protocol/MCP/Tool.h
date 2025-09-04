//===- Tool.h -------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_PROTOCOL_MCP_TOOL_H
#define LLDB_PLUGINS_PROTOCOL_MCP_TOOL_H

#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Tool.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <optional>

namespace lldb_private::mcp {

class CommandTool : public lldb_protocol::mcp::Tool {
public:
  using lldb_protocol::mcp::Tool::Tool;
  ~CommandTool() = default;

<<<<<<< HEAD
  virtual llvm::Expected<protocol::TextResult>
  Call(const protocol::ToolArguments &args) = 0;

  virtual std::optional<llvm::json::Value> GetSchema() const {
    return llvm::json::Object{{"type", "object"}};
  }

  protocol::ToolDefinition GetDefinition() const;

  const std::string &GetName() { return m_name; }

private:
  std::string m_name;
  std::string m_description;
};

class CommandTool : public mcp::Tool {
public:
  using mcp::Tool::Tool;
  ~CommandTool() = default;

  virtual llvm::Expected<protocol::TextResult>
  Call(const protocol::ToolArguments &args) override;

  virtual std::optional<llvm::json::Value> GetSchema() const override;
};

class DebuggerListTool : public mcp::Tool {
public:
  using mcp::Tool::Tool;
  ~DebuggerListTool() = default;

  virtual llvm::Expected<protocol::TextResult>
  Call(const protocol::ToolArguments &args) override;
};

=======
  llvm::Expected<lldb_protocol::mcp::CallToolResult>
  Call(const lldb_protocol::mcp::ToolArguments &args) override;

  std::optional<llvm::json::Value> GetSchema() const override;
};

>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
} // namespace lldb_private::mcp

#endif
