//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProtocolMCPTestUtilities.h"
#include "TestingSupport/Host/JSONTransportTestUtilities.h"
#include "TestingSupport/Host/PipeTestUtilities.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/JSONTransport.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "lldb/Host/Socket.h"
#include "lldb/Protocol/MCP/MCPError.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Resource.h"
#include "lldb/Protocol/MCP/Server.h"
#include "lldb/Protocol/MCP/Tool.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <chrono>
#include <condition_variable>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_protocol::mcp;

namespace {
class TestMCPTransport final : public MCPTransport {
public:
  TestMCPTransport(lldb::IOObjectSP in, lldb::IOObjectSP out)
      : lldb_protocol::mcp::MCPTransport(in, out, "unittest") {}

  using MCPTransport::Write;

  void Log(llvm::StringRef message) override {
    log_messages.emplace_back(message);
  }

  std::vector<std::string> log_messages;
};

class TestServer : public Server {
public:
  using Server::Server;
};

/// Test tool that returns it argument as text.
class TestTool : public Tool {
public:
  using Tool::Tool;

<<<<<<< HEAD
  virtual llvm::Expected<mcp::protocol::TextResult>
  Call(const ToolArguments &args) override {
=======
  llvm::Expected<CallToolResult> Call(const ToolArguments &args) override {
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
    std::string argument;
    if (const json::Object *args_obj =
            std::get<json::Value>(args).getAsObject()) {
      if (const json::Value *s = args_obj->get("arguments")) {
        argument = s->getAsString().value_or("");
      }
    }

    CallToolResult text_result;
    text_result.content.emplace_back(TextContent{{argument}});
    return text_result;
  }
};

class TestResourceProvider : public ResourceProvider {
  using ResourceProvider::ResourceProvider;

<<<<<<< HEAD
  virtual llvm::Expected<mcp::protocol::TextResult>
  Call(const ToolArguments &args) override {
=======
  std::vector<Resource> GetResources() const override {
    std::vector<Resource> resources;

    Resource resource;
    resource.uri = "lldb://foo/bar";
    resource.name = "name";
    resource.description = "description";
    resource.mimeType = "application/json";

    resources.push_back(resource);
    return resources;
  }

  llvm::Expected<ReadResourceResult>
  ReadResource(llvm::StringRef uri) const override {
    if (uri != "lldb://foo/bar")
      return llvm::make_error<UnsupportedURI>(uri.str());

    TextResourceContents contents;
    contents.uri = "lldb://foo/bar";
    contents.mimeType = "application/json";
    contents.text = "foobar";

    ReadResourceResult result;
    result.contents.push_back(contents);
    return result;
  }
};

/// Test tool that returns an error.
class ErrorTool : public Tool {
public:
  using Tool::Tool;

  llvm::Expected<CallToolResult> Call(const ToolArguments &args) override {
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
    return llvm::createStringError("error");
  }
};

/// Test tool that fails but doesn't return an error.
class FailTool : public Tool {
public:
  using Tool::Tool;

<<<<<<< HEAD
  virtual llvm::Expected<mcp::protocol::TextResult>
  Call(const ToolArguments &args) override {
    mcp::protocol::TextResult text_result;
    text_result.content.emplace_back(mcp::protocol::TextContent{{"failed"}});
=======
  llvm::Expected<CallToolResult> Call(const ToolArguments &args) override {
    CallToolResult text_result;
    text_result.content.emplace_back(TextContent{{"failed"}});
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
    text_result.isError = true;
    return text_result;
  }
};

class ProtocolServerMCPTest : public PipePairTest {
public:
  SubsystemRAII<FileSystem, HostInfo, Socket> subsystems;

  std::unique_ptr<TestMCPTransport> transport_up;
  std::unique_ptr<TestServer> server_up;
  MainLoop loop;
  MockMessageHandler<Request, Response, Notification> message_handler;

  llvm::Error Write(llvm::StringRef message) {
    llvm::Expected<json::Value> value = json::parse(message);
    if (!value)
      return value.takeError();
    return transport_up->Write(*value);
  }

  llvm::Error Write(json::Value value) { return transport_up->Write(value); }

  /// Run the transport MainLoop and return any messages received.
  llvm::Error
  Run(std::chrono::milliseconds timeout = std::chrono::milliseconds(200)) {
    loop.AddCallback([](MainLoopBase &loop) { loop.RequestTermination(); },
                     timeout);
    auto handle = transport_up->RegisterMessageHandler(loop, message_handler);
    if (!handle)
      return handle.takeError();

    return server_up->Run();
  }

  void SetUp() override {
    PipePairTest::SetUp();

<<<<<<< HEAD
    // Create & start the server.
    ProtocolServer::Connection connection;
    connection.protocol = Socket::SocketProtocol::ProtocolTcp;
    connection.name = llvm::formatv("{0}:0", k_localhost).str();
    m_server_up = std::make_unique<TestProtocolServerMCP>();
    m_server_up->AddTool(std::make_unique<TestTool>("test", "test tool"));
    ASSERT_THAT_ERROR(m_server_up->Start(connection), llvm::Succeeded());
=======
    transport_up = std::make_unique<TestMCPTransport>(
        std::make_shared<NativeFile>(input.GetReadFileDescriptor(),
                                     File::eOpenOptionReadOnly,
                                     NativeFile::Unowned),
        std::make_shared<NativeFile>(output.GetWriteFileDescriptor(),
                                     File::eOpenOptionWriteOnly,
                                     NativeFile::Unowned));
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

    server_up = std::make_unique<TestServer>(
        "lldb-mcp", "0.1.0",
        std::make_unique<TestMCPTransport>(
            std::make_shared<NativeFile>(output.GetReadFileDescriptor(),
                                         File::eOpenOptionReadOnly,
                                         NativeFile::Unowned),
            std::make_shared<NativeFile>(input.GetWriteFileDescriptor(),
                                         File::eOpenOptionWriteOnly,
                                         NativeFile::Unowned)),
        loop);
  }
};

template <typename T>
Request make_request(StringLiteral method, T &&params, Id id = 1) {
  return Request{id, method.str(), toJSON(std::forward<T>(params))};
}

template <typename T> Response make_response(T &&result, Id id = 1) {
  return Response{id, std::forward<T>(result)};
}

} // namespace

<<<<<<< HEAD
TEST_F(ProtocolServerMCPTest, Intialization) {
  llvm::StringLiteral request =
      R"json({"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"lldb-unit","version":"0.1.0"}},"jsonrpc":"2.0","id":0})json";
  llvm::StringLiteral response =
      R"json({"jsonrpc":"2.0","id":0,"result":{"capabilities":{"tools":{"listChanged":true}},"protocolVersion":"2024-11-05","serverInfo":{"name":"lldb-mcp","version":"0.1.0"}}})json";
=======
TEST_F(ProtocolServerMCPTest, Initialization) {
  Request request = make_request(
      "initialize", InitializeParams{/*protocolVersion=*/"2024-11-05",
                                     /*capabilities=*/{},
                                     /*clientInfo=*/{"lldb-unit", "0.1.0"}});
  Response response = make_response(
      InitializeResult{/*protocolVersion=*/"2024-11-05",
                       /*capabilities=*/{/*supportsToolsList=*/true},
                       /*serverInfo=*/{"lldb-mcp", "0.1.0"}});
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

  ASSERT_THAT_ERROR(Write(request), Succeeded());
  EXPECT_CALL(message_handler, Received(response));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, ToolsList) {
<<<<<<< HEAD
  llvm::StringLiteral request =
      R"json({"method":"tools/list","params":{},"jsonrpc":"2.0","id":1})json";
  llvm::StringLiteral response =
      R"json( {"id":1,"jsonrpc":"2.0","result":{"tools":[{"description":"test tool","inputSchema":{"type":"object"},"name":"test"},{"description":"List debugger instances with their debugger_id.","inputSchema":{"type":"object"},"name":"lldb_debugger_list"},{"description":"Run an lldb command.","inputSchema":{"properties":{"arguments":{"type":"string"},"debugger_id":{"type":"number"}},"required":["debugger_id"],"type":"object"},"name":"lldb_command"}]}})json";
=======
  server_up->AddTool(std::make_unique<TestTool>("test", "test tool"));

  Request request = make_request("tools/list", Void{}, /*id=*/"one");

  ToolDefinition test_tool;
  test_tool.name = "test";
  test_tool.description = "test tool";
  test_tool.inputSchema = json::Object{{"type", "object"}};

  Response response = make_response(ListToolsResult{{test_tool}}, /*id=*/"one");
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(response));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, ResourcesList) {
  server_up->AddResourceProvider(std::make_unique<TestResourceProvider>());

  Request request = make_request("resources/list", Void{});
  Response response = make_response(ListResourcesResult{
      {{/*uri=*/"lldb://foo/bar", /*name=*/"name",
        /*description=*/"description", /*mimeType=*/"application/json"}}});

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(response));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, ToolsCall) {
<<<<<<< HEAD
  llvm::StringLiteral request =
      R"json({"method":"tools/call","params":{"name":"test","arguments":{"arguments":"foo","debugger_id":0}},"jsonrpc":"2.0","id":11})json";
  llvm::StringLiteral response =
      R"json({"id":11,"jsonrpc":"2.0","result":{"content":[{"text":"foo","type":"text"}],"isError":false}})json";
=======
  server_up->AddTool(std::make_unique<TestTool>("test", "test tool"));

  Request request = make_request(
      "tools/call", CallToolParams{/*name=*/"test", /*arguments=*/json::Object{
                                       {"arguments", "foo"},
                                       {"debugger_id", 0},
                                   }});
  Response response = make_response(CallToolResult{{{/*text=*/"foo"}}});
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(response));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, ToolsCallError) {
  server_up->AddTool(std::make_unique<ErrorTool>("error", "error tool"));

<<<<<<< HEAD
  llvm::StringLiteral request =
      R"json({"method":"tools/call","params":{"name":"error","arguments":{"arguments":"foo","debugger_id":0}},"jsonrpc":"2.0","id":11})json";
  llvm::StringLiteral response =
      R"json({"error":{"code":-1,"message":"error"},"id":11,"jsonrpc":"2.0"})json";
=======
  Request request = make_request(
      "tools/call", CallToolParams{/*name=*/"error", /*arguments=*/json::Object{
                                       {"arguments", "foo"},
                                       {"debugger_id", 0},
                                   }});
  Response response =
      make_response(lldb_protocol::mcp::Error{eErrorCodeInternalError,
                                              /*message=*/"error"});
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(response));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, ToolsCallFail) {
  server_up->AddTool(std::make_unique<FailTool>("fail", "fail tool"));

<<<<<<< HEAD
  llvm::StringLiteral request =
      R"json({"method":"tools/call","params":{"name":"fail","arguments":{"arguments":"foo","debugger_id":0}},"jsonrpc":"2.0","id":11})json";
  llvm::StringLiteral response =
      R"json({"id":11,"jsonrpc":"2.0","result":{"content":[{"text":"failed","type":"text"}],"isError":true}})json";
=======
  Request request = make_request(
      "tools/call", CallToolParams{/*name=*/"fail", /*arguments=*/json::Object{
                                       {"arguments", "foo"},
                                       {"debugger_id", 0},
                                   }});
  Response response =
      make_response(CallToolResult{{{/*text=*/"failed"}}, /*isError=*/true});
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  EXPECT_CALL(message_handler, Received(response));
  EXPECT_THAT_ERROR(Run(), Succeeded());
}

TEST_F(ProtocolServerMCPTest, NotificationInitialized) {
  bool handler_called = false;
  std::condition_variable cv;

  server_up->AddNotificationHandler(
      "notifications/initialized",
      [&](const Notification &notification) { handler_called = true; });
  llvm::StringLiteral request =
      R"json({"method":"notifications/initialized","jsonrpc":"2.0"})json";

  ASSERT_THAT_ERROR(Write(request), llvm::Succeeded());
  EXPECT_THAT_ERROR(Run(), Succeeded());
  EXPECT_TRUE(handler_called);
}
