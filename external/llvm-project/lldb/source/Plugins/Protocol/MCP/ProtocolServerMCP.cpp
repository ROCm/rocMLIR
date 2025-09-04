//===- ProtocolServerMCP.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProtocolServerMCP.h"
#include "Resource.h"
#include "Tool.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Protocol/MCP/Server.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Threading.h"
#include <thread>

using namespace lldb_private;
using namespace lldb_private::mcp;
using namespace lldb_protocol::mcp;
using namespace llvm;

LLDB_PLUGIN_DEFINE(ProtocolServerMCP)

static constexpr llvm::StringLiteral kName = "lldb-mcp";
static constexpr llvm::StringLiteral kVersion = "0.1.0";

<<<<<<< HEAD
ProtocolServerMCP::ProtocolServerMCP() : ProtocolServer() {
  AddRequestHandler("initialize",
                    std::bind(&ProtocolServerMCP::InitializeHandler, this,
                              std::placeholders::_1));
  AddRequestHandler("tools/list",
                    std::bind(&ProtocolServerMCP::ToolsListHandler, this,
                              std::placeholders::_1));
  AddRequestHandler("tools/call",
                    std::bind(&ProtocolServerMCP::ToolsCallHandler, this,
                              std::placeholders::_1));
  AddNotificationHandler(
      "notifications/initialized", [](const protocol::Notification &) {
        LLDB_LOG(GetLog(LLDBLog::Host), "MCP initialization complete");
      });
  AddTool(
      std::make_unique<CommandTool>("lldb_command", "Run an lldb command."));
  AddTool(std::make_unique<DebuggerListTool>(
      "lldb_debugger_list", "List debugger instances with their debugger_id."));
}
=======
ProtocolServerMCP::ProtocolServerMCP() : ProtocolServer() {}
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

ProtocolServerMCP::~ProtocolServerMCP() { llvm::consumeError(Stop()); }

void ProtocolServerMCP::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void ProtocolServerMCP::Terminate() {
  if (llvm::Error error = ProtocolServer::Terminate())
    LLDB_LOG_ERROR(GetLog(LLDBLog::Host), std::move(error), "{0}");
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb::ProtocolServerUP ProtocolServerMCP::CreateInstance() {
  return std::make_unique<ProtocolServerMCP>();
}

llvm::StringRef ProtocolServerMCP::GetPluginDescriptionStatic() {
  return "MCP Server.";
}

void ProtocolServerMCP::Extend(lldb_protocol::mcp::Server &server) const {
  server.AddNotificationHandler("notifications/initialized",
                                [](const lldb_protocol::mcp::Notification &) {
                                  LLDB_LOG(GetLog(LLDBLog::Host),
                                           "MCP initialization complete");
                                });
  server.AddTool(
      std::make_unique<CommandTool>("lldb_command", "Run an lldb command."));
  server.AddResourceProvider(std::make_unique<DebuggerResourceProvider>());
}

void ProtocolServerMCP::AcceptCallback(std::unique_ptr<Socket> socket) {
  Log *log = GetLog(LLDBLog::Host);
  std::string client_name = llvm::formatv("client_{0}", m_instances.size() + 1);
  LLDB_LOG(log, "New MCP client connected: {0}", client_name);

  lldb::IOObjectSP io_sp = std::move(socket);
  auto transport_up = std::make_unique<lldb_protocol::mcp::MCPTransport>(
      io_sp, io_sp, std::move(client_name), [&](llvm::StringRef message) {
        LLDB_LOG(GetLog(LLDBLog::Host), "{0}", message);
      });
  auto instance_up = std::make_unique<lldb_protocol::mcp::Server>(
      std::string(kName), std::string(kVersion), std::move(transport_up),
      m_loop);
  Extend(*instance_up);
  llvm::Error error = instance_up->Run();
  if (error) {
    LLDB_LOG_ERROR(log, std::move(error), "Failed to run MCP server: {0}");
    return;
  }
  m_instances.push_back(std::move(instance_up));
}

llvm::Error ProtocolServerMCP::Start(ProtocolServer::Connection connection) {
  std::lock_guard<std::mutex> guard(m_mutex);

  if (m_running)
    return llvm::createStringError("the MCP server is already running");

  Status status;
  m_listener = Socket::Create(connection.protocol, status);
  if (status.Fail())
    return status.takeError();

  status = m_listener->Listen(connection.name, /*backlog=*/5);
  if (status.Fail())
    return status.takeError();

  auto handles =
      m_listener->Accept(m_loop, std::bind(&ProtocolServerMCP::AcceptCallback,
                                           this, std::placeholders::_1));
  if (llvm::Error error = handles.takeError())
    return error;

<<<<<<< HEAD
=======
  auto listening_uris = m_listener->GetListeningConnectionURI();
  if (listening_uris.empty())
    return createStringError("failed to get listening connections");
  std::string address =
      llvm::join(m_listener->GetListeningConnectionURI(), ", ");

  FileSpec user_lldb_dir = HostInfo::GetUserLLDBDir();

  Status error(llvm::sys::fs::create_directory(user_lldb_dir.GetPath()));
  if (error.Fail())
    return error.takeError();

  m_mcp_registry_entry_path = user_lldb_dir.CopyByAppendingPathComponent(
      formatv("lldb-mcp-{0}.json", getpid()).str());

  ServerInfo info;
  info.connection_uri = listening_uris[0];
  info.pid = getpid();

  std::string buf = formatv("{0}", toJSON(info)).str();
  size_t num_bytes = buf.size();

  const File::OpenOptions flags = File::eOpenOptionWriteOnly |
                                  File::eOpenOptionCanCreate |
                                  File::eOpenOptionTruncate;
  llvm::Expected<lldb::FileUP> file =
      FileSystem::Instance().Open(m_mcp_registry_entry_path, flags,
                                  lldb::eFilePermissionsFileDefault, false);
  if (!file)
    return file.takeError();
  if (llvm::Error error = (*file)->Write(buf.data(), num_bytes).takeError())
    return error;

>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
  m_running = true;
  m_listen_handlers = std::move(*handles);
  m_loop_thread = std::thread([=] {
    llvm::set_thread_name("protocol-server.mcp");
    m_loop.Run();
  });

  return llvm::Error::success();
}

llvm::Error ProtocolServerMCP::Stop() {
  {
<<<<<<< HEAD
    std::lock_guard<std::mutex> guard(m_server_mutex);
=======
    std::lock_guard<std::mutex> guard(m_mutex);
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
    if (!m_running)
      return createStringError("the MCP sever is not running");
    m_running = false;
  }

  if (!m_mcp_registry_entry_path.GetPath().empty())
    FileSystem::Instance().RemoveFile(m_mcp_registry_entry_path);
  m_mcp_registry_entry_path.Clear();

  // Stop the main loop.
  m_loop.AddPendingCallback(
      [](lldb_private::MainLoopBase &loop) { loop.RequestTermination(); });

  // Wait for the main loop to exit.
  if (m_loop_thread.joinable())
    m_loop_thread.join();

  return llvm::Error::success();
}
<<<<<<< HEAD

llvm::Expected<std::optional<protocol::Message>>
ProtocolServerMCP::HandleData(llvm::StringRef data) {
  auto message = llvm::json::parse<protocol::Message>(/*JSON=*/data);
  if (!message)
    return message.takeError();

  if (const protocol::Request *request =
          std::get_if<protocol::Request>(&(*message))) {
    llvm::Expected<protocol::Response> response = Handle(*request);

    // Handle failures by converting them into an Error message.
    if (!response) {
      protocol::Error protocol_error;
      llvm::handleAllErrors(
          response.takeError(),
          [&](const MCPError &err) { protocol_error = err.toProtcolError(); },
          [&](const llvm::ErrorInfoBase &err) {
            protocol_error.error.code = -1;
            protocol_error.error.message = err.message();
          });
      protocol_error.id = request->id;
      return protocol_error;
    }

    return *response;
  }

  if (const protocol::Notification *notification =
          std::get_if<protocol::Notification>(&(*message))) {
    Handle(*notification);
    return std::nullopt;
  }

  if (std::get_if<protocol::Error>(&(*message)))
    return llvm::createStringError("unexpected MCP message: error");

  if (std::get_if<protocol::Response>(&(*message)))
    return llvm::createStringError("unexpected MCP message: response");

  llvm_unreachable("all message types handled");
}

protocol::Capabilities ProtocolServerMCP::GetCapabilities() {
  protocol::Capabilities capabilities;
  capabilities.tools.listChanged = true;
  return capabilities;
}

void ProtocolServerMCP::AddTool(std::unique_ptr<Tool> tool) {
  std::lock_guard<std::mutex> guard(m_server_mutex);

  if (!tool)
    return;
  m_tools[tool->GetName()] = std::move(tool);
}

void ProtocolServerMCP::AddRequestHandler(llvm::StringRef method,
                                          RequestHandler handler) {
  std::lock_guard<std::mutex> guard(m_server_mutex);
  m_request_handlers[method] = std::move(handler);
}

void ProtocolServerMCP::AddNotificationHandler(llvm::StringRef method,
                                               NotificationHandler handler) {
  std::lock_guard<std::mutex> guard(m_server_mutex);
  m_notification_handlers[method] = std::move(handler);
}

llvm::Expected<protocol::Response>
ProtocolServerMCP::InitializeHandler(const protocol::Request &request) {
  protocol::Response response;
  response.result.emplace(llvm::json::Object{
      {"protocolVersion", protocol::kVersion},
      {"capabilities", GetCapabilities()},
      {"serverInfo",
       llvm::json::Object{{"name", kName}, {"version", kVersion}}}});
  return response;
}

llvm::Expected<protocol::Response>
ProtocolServerMCP::ToolsListHandler(const protocol::Request &request) {
  protocol::Response response;

  llvm::json::Array tools;
  for (const auto &tool : m_tools)
    tools.emplace_back(toJSON(tool.second->GetDefinition()));

  response.result.emplace(llvm::json::Object{{"tools", std::move(tools)}});

  return response;
}

llvm::Expected<protocol::Response>
ProtocolServerMCP::ToolsCallHandler(const protocol::Request &request) {
  protocol::Response response;

  if (!request.params)
    return llvm::createStringError("no tool parameters");

  const json::Object *param_obj = request.params->getAsObject();
  if (!param_obj)
    return llvm::createStringError("no tool parameters");

  const json::Value *name = param_obj->get("name");
  if (!name)
    return llvm::createStringError("no tool name");

  llvm::StringRef tool_name = name->getAsString().value_or("");
  if (tool_name.empty())
    return llvm::createStringError("no tool name");

  auto it = m_tools.find(tool_name);
  if (it == m_tools.end())
    return llvm::createStringError(llvm::formatv("no tool \"{0}\"", tool_name));

  protocol::ToolArguments tool_args;
  if (const json::Value *args = param_obj->get("arguments"))
    tool_args = *args;

  llvm::Expected<protocol::TextResult> text_result =
      it->second->Call(tool_args);
  if (!text_result)
    return text_result.takeError();

  response.result.emplace(toJSON(*text_result));

  return response;
}
=======
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
