Invoicing Completion

When all expected invoices for a unit have been generated, the system may mark the unit as:

INVOICING_COMPLETED

This status indicates that invoicing for that unit is considered finished.

Important edge case

New costs may still appear after invoicing completion.

Examples:

late construction changes

design modifications

regulatory requirements

unexpected structural work

When new costs are approved after invoicing completion, the system should create a separate invoice.

The system should warn administrators when this situation occurs.

Common mistake

Project managers sometimes assume that marking invoicing as completed prevents new invoices.

This is incorrect.

New invoices may still be created.
