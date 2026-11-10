"""An existing VPC on AWS ([aws] vpc_id, private_subnet_id, public_subnet_id,
s3_endpoint_id): the CIDRs read from AWS, the config rules, and the tfvars.
Run: python3 -m unittest discover -s tests"""
import _isolation  # noqa: F401  (first: the tests never read the real customer defaults)
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

from boxd_installer import config, tf
from boxd_installer.clouds import aws
from boxd_installer.util import InstallError

from test_machine_sizes import FakeState, aws_doc, validate

VPC, PRIV, PUB, VPCE = "vpc-0abc", "subnet-0priv", "subnet-0pub", "vpce-0s3"
PRIV_RTB, PUB_RTB = "rtb-0priv", "rtb-0pub"

class FakeEc2:
    """aws ec2 describe-* for a VPC 10.60.0.0/16 in us-east-1 (the example's
    region), its subnets and S3 gateway endpoint."""
    def __init__(self, subnet_az="us-east-1a", subnet_vpc=VPC, endpoint_tables=(PRIV_RTB, PUB_RTB),
                 endpoint_service="com.amazonaws.us-east-1.s3", missing=(), denied=()):
        self.calls, self.missing, self.denied = [], missing, denied
        self.subnets = {
            PRIV: {"SubnetId": PRIV, "VpcId": subnet_vpc, "CidrBlock": "10.60.4.0/22",
                   "AvailabilityZone": subnet_az},
            PUB: {"SubnetId": PUB, "VpcId": VPC, "CidrBlock": "10.60.0.0/24",
                  "AvailabilityZone": "us-east-1a"},
        }
        self.endpoint = {"VpcEndpointId": VPCE, "VpcId": VPC, "VpcEndpointType": "Gateway",
                         "ServiceName": endpoint_service, "RouteTableIds": list(endpoint_tables)}

    def __call__(self, region, op, *args):
        self.calls.append((region, op, *args))
        if set(args) & set(self.denied):
            raise InstallError(f"command failed (254): aws ec2 {op} ...\nAn error occurred "
                               f"(UnauthorizedOperation) when calling the {op} operation")
        if set(args) & set(self.missing):
            return {"Vpcs": [], "Subnets": [], "VpcEndpoints": []}
        if op == "describe-vpcs":
            return {"Vpcs": [{"VpcId": VPC, "CidrBlock": "10.60.0.0/16"}]}
        if op == "describe-subnets":
            return {"Subnets": [self.subnets[s] for s in args[1:]]}
        if op == "describe-vpc-endpoints":
            return {"VpcEndpoints": [self.endpoint]}
        if op == "describe-route-tables":
            return {"RouteTables": [
                {"RouteTableId": PRIV_RTB, "Associations": [{"SubnetId": PRIV}]},
                {"RouteTableId": PUB_RTB, "Associations": [{"SubnetId": PUB}]},
                {"RouteTableId": "rtb-0main", "Associations": [{"Main": True}]},
            ]}
        raise AssertionError(op)

def existing_doc(ingress="private", **extra) -> dict:
    """examples/aws/cluster.toml moved into the existing VPC: both nodes at
    the top of the private subnet (10.60.4.0/22)."""
    doc = aws_doc()
    for k in aws.NETWORK_CIDRS:
        del doc["aws"][k]
    doc["aws"].update({"vpc_id": VPC, "private_subnet_id": PRIV, **extra})
    if ingress == "private":
        doc["aws"]["ingress"] = "private"
        doc["acme"] = {"ca": "local"}
        doc["nodes"][0]["internal_ip"] = "10.60.7.250"
    else:
        doc["aws"].setdefault("public_subnet_id", PUB)
        doc["nodes"][0]["internal_ip"] = "10.60.0.250"
    doc["nodes"][1]["internal_ip"] = "10.60.7.251"
    return doc

def resolve(doc: dict, ec2=None) -> str:
    """existing_network then validation, as config.load runs them: errors as
    one string ("" when valid)."""
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            aws.existing_network(doc["aws"], ec2 or FakeEc2())
    except InstallError as e:
        return str(e)
    return validate(doc)

class ExistingNetwork(unittest.TestCase):
    def test_private_reads_the_cidrs(self):
        doc, ec2 = existing_doc(), FakeEc2()
        self.assertEqual(resolve(doc, ec2), "")
        self.assertEqual(doc["aws"]["vpc_cidr"], "10.60.0.0/16")
        self.assertEqual(doc["aws"]["private_subnet_cidr"], "10.60.4.0/22")
        self.assertNotIn("public_subnet_cidr", doc["aws"])
        self.assertEqual([c[:2] for c in ec2.calls], [("us-east-1", "describe-vpcs"),
                                                      ("us-east-1", "describe-subnets")])
        # The VPC is what the proxy admits by default, and boxd's private network.
        self.assertEqual(aws.ingress_cidrs(doc["aws"]), ["10.60.0.0/16"])
        self.assertEqual(config.Config(doc).vpc_cidr, "10.60.0.0/16")

    def test_public_reads_both_subnets(self):
        doc = existing_doc("public")
        self.assertEqual(resolve(doc), "")
        self.assertEqual(doc["aws"]["public_subnet_cidr"], "10.60.0.0/24")
        self.assertEqual(aws.ingress_cidrs(doc["aws"]), ["0.0.0.0/0"])

    def test_refuses_cidrs(self):
        for k, v in (("vpc_cidr", "10.60.0.0/16"), ("private_subnet_cidr", "10.60.4.0/22"),
                     ("public_subnet_cidr", "10.60.0.0/24")):
            doc = existing_doc(**{k: v})
            self.assertIn(f"[aws].{k} comes from vpc_id's VPC and subnets: remove it",
                          resolve(doc), k)

    def test_private_needs_only_the_private_subnet(self):
        doc = existing_doc()
        del doc["aws"]["private_subnet_id"]
        self.assertIn("[aws].vpc_id needs private_subnet_id", resolve(doc))
        self.assertIn('public_subnet_id is only for ingress = "public"',
                      resolve(existing_doc(public_subnet_id=PUB)))

    def test_needs_region_and_zone(self):
        for k in ("region", "availability_zone"):
            doc = existing_doc()
            del doc["aws"][k]
            self.assertIn(f"[aws].vpc_id needs {k}", resolve(doc), k)

    def test_public_needs_the_public_subnet(self):
        doc = existing_doc("public")
        del doc["aws"]["public_subnet_id"]
        self.assertIn('needs public_subnet_id with ingress = "public"', resolve(doc))

    def test_ids_are_strings(self):
        self.assertIn("[aws].private_subnet_id must be a quoted id string",
                      resolve(existing_doc(private_subnet_id=7)))

    def test_node_must_be_in_its_subnet(self):
        doc = existing_doc()
        doc["nodes"][1]["internal_ip"] = "10.60.8.10"
        self.assertIn("internal_ip 10.60.8.10 must be in the private subnet 10.60.4.0/22",
                      resolve(doc))
        doc = existing_doc("public")
        doc["nodes"][0]["internal_ip"] = "10.60.7.250"
        self.assertIn("internal_ip 10.60.7.250 must be in the public subnet 10.60.0.0/24",
                      resolve(doc))
        doc = existing_doc()
        doc["nodes"][1]["internal_ip"] = "10.60.7.255"
        self.assertIn("is AWS-reserved", resolve(doc))

    def test_subnet_in_another_vpc_or_zone(self):
        self.assertIn(f"private_subnet_id {PRIV} is in vpc-0other, not vpc_id {VPC}",
                      resolve(existing_doc(), FakeEc2(subnet_vpc="vpc-0other")))
        self.assertIn(f"private_subnet_id {PRIV} is in us-east-1b, not availability_zone us-east-1a",
                      resolve(existing_doc(), FakeEc2(subnet_az="us-east-1b")))

    def test_ids_need_vpc_id(self):
        doc = aws_doc()
        doc["aws"]["private_subnet_id"] = PRIV
        self.assertIn("[aws].private_subnet_id is for an existing VPC: it needs vpc_id",
                      validate(doc))

    def test_load_reads_the_vpc(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "examples", "aws",
                               "cluster.toml")) as fh:
            doc_toml = fh.read()
        for line in ('vpc_cidr            = "10.70.0.0/16"\n',
                     'public_subnet_cidr  = "10.70.0.0/24"\n',
                     'private_subnet_cidr = "10.70.1.0/24"\n'):
            doc_toml = doc_toml.replace(line, "")
        doc_toml = doc_toml.replace("[aws]\n", f'[aws]\nvpc_id = "{VPC}"\n'
                                    f'public_subnet_id = "{PUB}"\nprivate_subnet_id = "{PRIV}"\n')
        doc_toml = doc_toml.replace('"10.70.0.10"', '"10.60.0.250"').replace('"10.70.1.10"',
                                                                           '"10.60.7.251"')
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cluster.toml")
            with open(path, "w") as fh:
                fh.write(doc_toml)
            with mock.patch.object(aws, "_describe", FakeEc2()), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                cfg = config.load(path)
        self.assertEqual(cfg.vpc_cidr, "10.60.0.0/16")
        self.assertIn(f"[aws] vpc_id {VPC}: VPC 10.60.0.0/16 (resolver 10.60.0.2)", err.getvalue())

class Lookups(unittest.TestCase):
    def test_not_found(self):
        for key, ident, extra in (("vpc_id", VPC, {}), ("private_subnet_id", PRIV, {}),
                                  ("s3_endpoint_id", VPCE,
                                   {"bucket_access": "vpc", "s3_endpoint_id": VPCE})):
            err = resolve(existing_doc(**extra), FakeEc2(missing=(ident,)))
            self.assertIn(f"[aws].{key} {ident}: not found in us-east-1", err, key)

    def test_denied_or_unknown_id(self):
        err = resolve(existing_doc("public"), FakeEc2(denied=(PUB,)))
        self.assertIn(f"[aws].public_subnet_id {PUB}: AWS can't find it in us-east-1, or this "
                      "login may not describe it (An error occurred (UnauthorizedOperation)", err)

class NetworkSwitch(unittest.TestCase):
    """A cluster that exists stays in the kind of VPC it has."""
    def setUp(self):
        self.created = config.Config(aws_doc())
        doc = existing_doc()
        resolve(doc)
        self.existing = config.Config(doc)

    def test_same_mode_or_new_cluster_passes(self):
        for mode in (None, "created"):
            aws.check_network_switch(self.created, mode)
        for mode in (None, "existing"):
            aws.check_network_switch(self.existing, mode)

    def test_created_to_vpc_id_refused(self):
        with self.assertRaisesRegex(InstallError, "runs in the VPC apply made, and cluster.toml "
                                    "now sets \\[aws\\] vpc_id"):
            aws.check_network_switch(self.existing, "created")

    def test_vpc_id_to_created_refused(self):
        with self.assertRaisesRegex(InstallError, "runs in an existing VPC .* no longer sets it"):
            aws.check_network_switch(self.created, "existing")

    def test_state_network(self):
        made = {"resources": [{"mode": "managed", "type": "aws_vpc", "name": "vpc",
                               "instances": [{"index_key": 0}]}]}
        old = {"resources": [{"mode": "managed", "type": "aws_vpc", "name": "vpc",
                              "instances": [{}]}]}
        given = {"resources": [{"mode": "data", "type": "aws_vpc", "name": "existing",
                                "instances": [{"index_key": 0}]},
                               {"mode": "managed", "type": "aws_instance", "name": "node",
                                "instances": [{"index_key": "w-0"}]}]}
        self.assertEqual(tf.state_network(made), "created")
        self.assertEqual(tf.state_network(old), "created")
        self.assertEqual(tf.state_network(given), "existing")
        self.assertIsNone(tf.state_network({}))
        self.assertIsNone(tf.state_network({"resources": [
            {"mode": "managed", "type": "aws_vpc", "name": "vpc", "instances": []}]}))

class S3Endpoint(unittest.TestCase):
    def test_bucket_access_vpc_needs_the_endpoint(self):
        self.assertIn('bucket_access = "vpc" with vpc_id needs s3_endpoint_id',
                      resolve(existing_doc(bucket_access="vpc")))
        self.assertIn('s3_endpoint_id is only for bucket_access = "vpc"',
                      resolve(existing_doc(s3_endpoint_id=VPCE)))

    def test_endpoint_on_every_cluster_subnet(self):
        self.assertEqual(resolve(existing_doc(bucket_access="vpc", s3_endpoint_id=VPCE)), "")
        self.assertEqual(resolve(existing_doc("public", bucket_access="vpc",
                                              s3_endpoint_id=VPCE)), "")
        err = resolve(existing_doc("public", bucket_access="vpc", s3_endpoint_id=VPCE),
                      FakeEc2(endpoint_tables=(PRIV_RTB,)))
        self.assertIn(f"{VPCE} is not on {PUB}'s route table {PUB_RTB}", err)

    def test_endpoint_must_be_s3_gateway(self):
        err = resolve(existing_doc(bucket_access="vpc", s3_endpoint_id=VPCE),
                      FakeEc2(endpoint_service="com.amazonaws.us-east-1.dynamodb"))
        self.assertIn(f"{VPCE} must be an S3 gateway endpoint", err)

def tfvars(doc: dict) -> dict:
    doc["aws"]["ami"] = "ami-0123456789abcdef0"  # skip the SSM lookup
    return tf.build_tfvars(config.Config(doc), FakeState())

class Tfvars(unittest.TestCase):
    def test_existing_vpc_passes_ids_not_cidrs(self):
        doc = existing_doc(bucket_access="vpc", s3_endpoint_id=VPCE)
        self.assertEqual(resolve(doc), "")
        tv = tfvars(doc)
        self.assertEqual({k: tv[k] for k in aws.EXISTING_VPC_KEYS},
                         {"vpc_id": VPC, "private_subnet_id": PRIV, "public_subnet_id": None,
                          "s3_endpoint_id": VPCE})
        for k in aws.NETWORK_CIDRS:
            self.assertNotIn(k, tv)
        self.assertEqual(tv["ingress_cidrs"], ["10.60.0.0/16"])

    def test_default_unchanged(self):
        tv = tfvars(aws_doc())
        self.assertEqual({k: tv[k] for k in aws.NETWORK_CIDRS},
                         {"vpc_cidr": "10.70.0.0/16", "public_subnet_cidr": "10.70.0.0/24",
                          "private_subnet_cidr": "10.70.1.0/24"})
        for k in aws.EXISTING_VPC_KEYS:
            self.assertNotIn(k, tv)

if __name__ == "__main__":
    unittest.main()
