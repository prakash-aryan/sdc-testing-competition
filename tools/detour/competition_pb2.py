# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: competition.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'competition.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11\x63ompetition.proto\"\x07\n\x05\x45mpty\"\x19\n\tNameReply\x12\x0c\n\x04name\x18\x01 \x01(\t\";\n\x06Oracle\x12\x1e\n\x08testCase\x18\x01 \x01(\x0b\x32\x0c.SDCTestCase\x12\x11\n\thasFailed\x18\x02 \x01(\x08\"=\n\x0bSDCTestCase\x12\x0e\n\x06testId\x18\x01 \x01(\t\x12\x1e\n\nroadPoints\x18\x02 \x03(\x0b\x32\n.RoadPoint\"9\n\tRoadPoint\x12\x16\n\x0esequenceNumber\x18\x01 \x01(\x03\x12\t\n\x01x\x18\x02 \x01(\x02\x12\t\n\x01y\x18\x03 \x01(\x02\"!\n\x13InitializationReply\x12\n\n\x02ok\x18\x01 \x01(\x08\" \n\x0eSelectionReply\x12\x0e\n\x06testId\x18\x01 \x01(\t2\x8f\x01\n\x0f\x43ompetitionTool\x12\x1c\n\x04Name\x12\x06.Empty\x1a\n.NameReply\"\x00\x12/\n\nInitialize\x12\x07.Oracle\x1a\x14.InitializationReply\"\x00(\x01\x12-\n\x06Select\x12\x0c.SDCTestCase\x1a\x0f.SelectionReply\"\x00(\x01\x30\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'competition_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_EMPTY']._serialized_start=21
  _globals['_EMPTY']._serialized_end=28
  _globals['_NAMEREPLY']._serialized_start=30
  _globals['_NAMEREPLY']._serialized_end=55
  _globals['_ORACLE']._serialized_start=57
  _globals['_ORACLE']._serialized_end=116
  _globals['_SDCTESTCASE']._serialized_start=118
  _globals['_SDCTESTCASE']._serialized_end=179
  _globals['_ROADPOINT']._serialized_start=181
  _globals['_ROADPOINT']._serialized_end=238
  _globals['_INITIALIZATIONREPLY']._serialized_start=240
  _globals['_INITIALIZATIONREPLY']._serialized_end=273
  _globals['_SELECTIONREPLY']._serialized_start=275
  _globals['_SELECTIONREPLY']._serialized_end=307
  _globals['_COMPETITIONTOOL']._serialized_start=310
  _globals['_COMPETITIONTOOL']._serialized_end=453
# @@protoc_insertion_point(module_scope)
